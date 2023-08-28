from click import command
import torch
import threading
import time
from speechbrain.pretrained import EncoderClassifier
from google.cloud import speech
from numpy import mean
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import os
import struct
import torchaudio
import eventlet
from jiwer import cer
from functools import reduce
eventlet.monkey_patch()
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOOGLE_APPLICATION_CREDENTIALS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "구글 KEY"
)
SAMPLE_RATE = 16000
SAMPLE_SIZE = 4  # int(2) -> float(4) RAW 처리용
BUF_TIME = 5
BUF_SIZE = SAMPLE_RATE * BUF_TIME
TICK_TIME = 3
THRESHOLD = 0.25
LOUDNESS_THRESHOLD = 0.005
LOUDNESS_SAMPLE_TIME = 0.1
STT_MIN_TIME = 1
STT_LOUDNESS_SAMPLE_TIME = 1
STT_BUF_TIME = 10
STT_SAMPLE_SIZE = 2
STT_BUF_SIZE = SAMPLE_RATE * STT_BUF_TIME
STT_MARGIN_TIME = 0.3
COMMAND_TIMEOUT_SEC = 18
MIN_EMB_SAMPLE_SEC = 10
MAX_FAIL_COUNT = 3


similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
classifier = EncoderClassifier.from_hparams(
    source=os.path.join(
        BASE_DIR, "pretrained_models\\SpeakerRecognition-8f6f7fdaa9628acf73e21ad1f99d5f83"))
lock = threading.Lock()
connections = {}
profiles = []
speechClient = speech.SpeechClient.from_service_account_json(
    GOOGLE_APPLICATION_CREDENTIALS)
recognitionConfig = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=SAMPLE_RATE,
    language_code="ko-KR",
)


def get_emb_by_pcm(pcm, extend=True):
    if extend:
        pcm = pcm[:]
        pcm_len = len(pcm)

        if pcm_len:
            for i in range(round(MIN_EMB_SAMPLE_SEC * SAMPLE_RATE) - pcm_len):
                i = i % pcm_len
                pcm.append(pcm[i])
    signal = torch.Tensor(pcm)
    waveform = classifier.audio_normalizer(signal, SAMPLE_RATE)
    batch = waveform.unsqueeze(0)
    emb = classifier.encode_batch(batch, normalize=True)

    return emb


def password_cv(dataset):
    dataset_len = len(dataset)
    error = [0 for _ in range(dataset_len)]
    for i in range(dataset_len):
        for j in range(dataset_len):
            if i == j:
                continue
            error[i] += cer(dataset[i][0], dataset[j][0])

    error = list(map(lambda x: x / (dataset_len - 1), error))

    min_error_idx = 0
    for i in range(dataset_len):
        if error[i] < error[min_error_idx]:
            min_error_idx = i

    return min_error_idx, error[min_error_idx]


def create_profile(profile):
    profiles.append(profile)
    with open("profiles.json", "w") as f:
        json.dump(profiles, f)


class Connection:
    def __init__(self, socketio, sid):
        self.socketio = socketio
        self.sid = sid
        self.profile = None
        self.buf = []
        self.stt_buf = []
        self.last_tick = 0
        self.stt_last_tick = 0
        self.score = 0
        self.task = "HOME"
        self.task_data = None
        self.command_timer = None
        self.stt_in_progress = False
        self.auth_fail_count = 0

    def set_profile(self, profile):
        self.profile = profile
        self.emb_tensor = torch.Tensor(profile["emb"])

    def clear_profile(self):
        self.profile = None
        self.emb_tensor = None

    def set_timeout(self):
        self.clear_timeout()
        self.command_timer = eventlet.spawn_after(
            COMMAND_TIMEOUT_SEC, self.command_timeout)

    def clear_timeout(self):
        if self.command_timer is not None:
            self.command_timer.cancel()
            self.command_timer = None

    def command_timeout(self):
        self.task = "HOME"
        self.socketio.emit("message", "입력시간 초과",
                           to=self.sid)
        self.socketio.emit(
            "play_audio", {"file": "timeout"}, to=self.sid)

    def auth_check(self, emb):
        profile_emb = self.emb_tensor
        score = similarity(profile_emb, emb).item()
        prediction = THRESHOLD < score
        print(f"score: {score}, prediction: {prediction}")
        self.socketio.emit("voice_recognition", {
            "score": score, "prediction": prediction}, to=self.sid)
        return prediction


class App:
    def __init__(self, port=8484):
        self.port = port
        self.flask = Flask('__name__',
                           static_url_path='',
                           static_folder='static',
                           template_folder='templates')
        self.flask.config['SECRET_KEY'] = 'test'
        self.flask.config['transports'] = 'websocket'
        self.socketio = SocketIO(self.flask)
        self.thread = None

        self.flask.route('/')(self.index)
        self.socketio.on('connect')(self.on_connect)
        self.socketio.on('audio_data')(self.audio_data)
        self.socketio.on('stop')(self.on_stop)

    def run(self):
        # self.flask.run()
        self.socketio.run(self.flask, host='0.0.0.0',
                          port=self.port, debug=True)

    def index(self):
        return render_template('index.html')

    def on_connect(self):
        sid = request.sid
        connections[sid] = Connection(self.socketio, sid)
        self.socketio.emit("connected", {"sid": sid}, to=sid)
        self.socketio.emit("play_audio", {"file": "home"}, to=sid)
        print("connected:", sid)

    def command_proc(self, connection, text, wave):
        sid = connection.sid

        if connection.task == "HOME":
            if text == "사용자 등록":
                print("사용자 등록 작업", sid)
                connection.task = "CREATE_PROFILE"
                # 페이즈, 프로필 이름, [비밀번호, wave, emb]
                connection.task_data = [1, "", []]
                self.socketio.emit(
                    "play_audio", {"file": "createProfile1"}, to=sid)
                connection.set_timeout()
            elif text == "사용자 선택":
                print("사용자 선택 작업")
                connection.task = "SELECT_PROFILE"
                self.socketio.emit("message", "선택할 사용자 이름을 말해주세요",
                                   to=sid)
                self.socketio.emit(
                    "play_audio", {"file": "selectProfile"}, to=sid)
                connection.set_timeout()
        elif connection.task == "CREATE_PROFILE":
            if connection.task_data[0] == 1:
                connection.task_data[1] = text
                connection.task_data[0] = 2
                self.socketio.emit(
                    "play_audio", {"file": "createProfile2"}, to=sid)
                connection.set_timeout()
            elif connection.task_data[0] == 2:
                if connection.task_data[1] == text:
                    for profile in profiles:
                        if profile["name"] == text:
                            self.socketio.emit("play_audio", {
                                "file": "existProfile"}, to=sid)
                            self.socketio.emit(
                                "play_audio", {"file": "home"}, to=sid)
                            connection.task = "HOME"
                            connection.clear_timeout()
                            break
                    else:
                        connection.task_data[0] = 3
                        self.socketio.emit(
                            "message", "프로필: " + text, to=sid)
                        self.socketio.emit(
                            "play_audio", {"file": "password1"}, to=sid)
                        connection.set_timeout()
                else:
                    self.socketio.emit(
                        "play_audio", {"file": "profileNameMismatch"}, to=sid)
                    self.socketio.emit("play_audio", {"file": "home"}, to=sid)
                    connection.task = "HOME"
                    connection.clear_timeout()
            elif connection.task_data[0] in [3, 4]:
                emb = get_emb_by_pcm(wave)
                connection.task_data[2].append([text, wave, emb])
                connection.task_data[0] += 1
                self.socketio.emit("play_audio", {
                    "file": f"password{connection.task_data[0] - 3}"}, to=sid)
                connection.set_timeout()
            elif connection.task_data[0] == 5:
                emb = get_emb_by_pcm(wave)
                connection.task_data[2].append([text, wave, emb])
                idx, error = password_cv(connection.task_data[2])
                if 0.2 < error:
                    self.socketio.emit(
                        "play_audio", {"file": "passwordMismatch"}, to=sid)
                    self.socketio.emit("play_audio", {"file": "home"}, to=sid)
                    connection.task = "HOME"
                else:
                    name = connection.task_data[1]
                    password = connection.task_data[2][idx][0]
                    profile_wave = reduce(
                        lambda acc, cur: acc + cur[1], connection.task_data[2], [])
                    profile_emb = get_emb_by_pcm(profile_wave)
                    profile_emb_list = profile_emb.data.cpu().numpy()[
                        0].tolist()

                    profile = {
                        "name": name,
                        "password": password,
                        "emb": profile_emb_list
                    }
                    create_profile(profile)

                    connection.set_profile(profile)
                    self.socketio.emit(
                        "play_audio", {"file": "profileCreated"}, to=sid)
                    self.socketio.emit("play_audio", {"file": "home"}, to=sid)
                    connection.task = "HOME"
                connection.clear_timeout()
        elif connection.task == "SELECT_PROFILE":
            for profile in profiles:
                if profile["name"] == text:
                    connections[sid].profile = profile
                    connections[sid].emb_tensor = torch.Tensor(profile["emb"])
                    self.socketio.emit(
                        "play_audio", {"file": "profileSelected"}, to=sid)
                    connection.task = "PROC"
                    break
            else:
                self.socketio.emit(
                    "play_audio", {"file": "profileNotFound"}, to=sid)
                self.socketio.emit("play_audio", {"file": "home"}, to=sid)
                connection.task = "HOME"
            connection.clear_timeout()
        elif connection.task == "PROC":
            COMMAND_LIST = list(map(lambda x: x.replace(" ", ""), [
                "베스트 몇 명",
                "테스트 명령",
                "베스트 명령",
                "데스크 명령",
                "명령어 수행"
            ]))
            emb = get_emb_by_pcm(wave)

            if text.replace(" ", "") in COMMAND_LIST:
                prediction = connection.auth_check(emb)
                if prediction:
                    self.socketio.emit(
                        "message", "COMMAND 수행: " + text, to=sid)
                    self.socketio.emit(
                        "play_audio", {"file": "commandExecuted"}, to=sid)
                    connection.auth_fail_count = 0
                else:
                    connection.auth_fail_count += 1
                    if MAX_FAIL_COUNT < connection.auth_fail_count:
                        connection.clear_profile()
                        self.socketio.emit(
                            "play_audio", {"file": "sessionExpired"}, to=sid)
                        self.socketio.emit(
                            "play_audio", {"file": "home"}, to=sid)
                        connection.task = "HOME"
                    else:
                        self.socketio.emit(
                            "message", f"성문 인증 실패: {connection.auth_fail_count}/{MAX_FAIL_COUNT}", to=sid)
                        self.socketio.emit("play_audio", {
                            "file": "verificationFailed"}, to=sid)

    def stt_recognize(self, connection, wave):
        print("RECOGNIZE...")
        text = ""
        audio = speech.RecognitionAudio(
            content=struct.pack(f"{len(wave)}h", *wave))
        response = speechClient.recognize(
            config=recognitionConfig, audio=audio)

        if response.results:
            result = response.results[0]
            text = result.alternatives[0].transcript

            self.command_proc(connection, text, wave)

        connection.stt_in_progress = False
        print("TEXT:", text)

        return text

    def stt_proc(self, connection, data):
        if connection.stt_in_progress:
            return
        wave = struct.unpack(f'{int(len(data) / 2)}h', data)
        connection.stt_buf += wave

        if SAMPLE_RATE * STT_LOUDNESS_SAMPLE_TIME <= len(connection.stt_buf):
            loudness_sample = connection.stt_buf[-int(
                SAMPLE_RATE * STT_LOUDNESS_SAMPLE_TIME):]
            loudness = mean([abs(x / 32767) for x in loudness_sample])

            if loudness < LOUDNESS_THRESHOLD:
                margin = max(
                    1, int(len(loudness_sample) - (SAMPLE_RATE * STT_MARGIN_TIME)))
                stt_wave = connection.stt_buf[:-margin]
                connection.stt_buf = []

                if SAMPLE_RATE * STT_MIN_TIME <= len(stt_wave):
                    # print("CUT")
                    connection.stt_in_progress = True
                    self.socketio.start_background_task(
                        self.stt_recognize, connection, stt_wave)
                # else:
                #     print("DROP")

            elif STT_BUF_SIZE < len(connection.stt_buf):
                connection.stt_buf = []
                # print("FULL DROP")

    def verification_proc(self, connection, data):
        if not connection.profile:
            return

        unpack_data = struct.unpack('128h', data)
        wave = [pcm / 32767 for pcm in unpack_data]
        connection.buf += wave

        loudness_sample = connection.buf[-int(
            SAMPLE_RATE * LOUDNESS_SAMPLE_TIME):]
        loudness = mean([abs(x) for x in loudness_sample])
        if loudness < LOUDNESS_THRESHOLD:
            connection.buf = connection.buf[:-len(loudness_sample)]

        if BUF_SIZE < len(connection.buf):
            connection.buf = connection.buf[-BUF_SIZE:]

        if BUF_SIZE <= len(connection.buf) or 8 <= len(connection.buf):
            if connection.last_tick + TICK_TIME <= time.time():
                profile_emb = connection.emb_tensor
                emb = get_emb_by_pcm(connection.buf)
                score = similarity(profile_emb, emb).item()
                prediction = THRESHOLD < score

                connection.score = score
                connection.last_tick = time.time()

                self.socketio.emit("voice_recognition", {
                    "score": score, "prediction": prediction}, to=connection.sid)

    def audio_data(self, data):
        sid = request.sid
        connection = connections[sid]

        if type(data) is not bytes:
            print("########## type(data) is not bytes")
            print(type(data))
            print(data)
            return

        self.stt_proc(connection, data)
        # self.verification_proc(connection, data)

    def on_stop(self):
        sid = request.sid
        connection = connections[sid]

        if not connection.stt_in_progress:
            stt_wave = connection.stt_buf
            if SAMPLE_RATE * STT_MIN_TIME <= len(stt_wave):
                connection.stt_in_progress = True

                self.socketio.start_background_task(
                    self.stt_recognize, connection, stt_wave)

        connection.stt_buf = []


def main():
    global profiles
    if os.path.exists("profiles.json"):
        with open("profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)

    app = App()
    app.run()


main()
