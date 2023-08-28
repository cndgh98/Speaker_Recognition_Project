io({ transports: ["websocket"] }); //, upgrade: false
// const socket = io.connect("http://127.0.0.1:8080");
let socket = null;
let processor = null;
let sid = null;
let loudness = 0;
let silenceStart = 0;
let userScore = 0;
let audioBuffer = [];
let PLAY = false;

const LOUDNESS_THRESHOLD = 0.005;
const SILENCE_TIME_MS = 3000;

let audio = new Audio();

// 그래프 속성값
const graphConfig = {
  loudnessChart: {
    millisPerPixel: 20,
    grid: {
      strokeStyle: "#000000",
    },
    labels: {
      disabled: true,
    },
    minValue: 0,
    maxValue: 0.5,
  },
  scoreChart: {
    millisPerPixel: 20,
    grid: {
      strokeStyle: "#000000",
    },
    labels: {
      disabled: true,
    },
    minValue: 0,
    maxValue: 0.5,
  },
  series: {
    lineWidth: 2,
    strokeStyle: "#00ff00",
  },
  delay: 1000,
};

const btn = document.getElementById("record-btn");

// 그래프에 그릴 데이터 저장하는 배열
let l_series = new TimeSeries();
let s_series = new TimeSeries();

// 그래프 생성
let l_graph = new SmoothieChart(graphConfig["loudnessChart"]);
// 데이터와 그래프 연ㅕ
l_graph.addTimeSeries(l_series, graphConfig["series"]);
// 캔버스와 그래프 연결
l_graph.streamTo(
  document.getElementById("loudness-graph"),
  graphConfig["delay"]
);

let s_graph = new SmoothieChart(graphConfig["scoreChart"]);
s_graph.addTimeSeries(s_series, graphConfig["series"]);
s_graph.streamTo(document.getElementById("score-graph"), graphConfig["delay"]);

function initHandler(params) {
  socket.on("connect", function () {
    socket.sendBuffer = [];
  });

  // 여기까지
  socket.on("connected", function ({ sid }) {
    console.log("sid:", sid);
  });

  socket.on("voice_recognition", function ({ score, prediction }) {
    userScore = score;
  });

  socket.on("play_audio", function (e) {
    audioBuffer.push(e["file"]);
    playAudio();
  });

  socket.on("message", function (message) {
    console.log("message: " + message);
  });
}

// 버튼 on/off 처리 로직
function voiceToggle() {
  btn.classList.toggle("on");

  if (btn.classList.contains("on")) {
    if (!socket) {
      socket = io.connect(null, { port: 5000, rememberTransport: false });
      initHandler();
    }
    processor.port.postMessage({ event: "toggle", data: true });
  } else {
    userScore = 0;
    socket.disconnect();
    socket = null;
    processor.port.postMessage({ event: "toggle", data: false });
  }
}

function playAudio(fileName) {
  if (!PLAY) {
    PLAY = true;
    file = audioBuffer.shift();
    processor.port.postMessage({ event: "toggle", data: false });
    audio.src = ` /audio/${file}.mp3`;
    audio.play();
  }
}

function checkAudio() {
  PLAY = false;

  if (audioBuffer && audioBuffer.length) {
    playAudio();
  } else {
    processor.port.postMessage({ event: "toggle", data: true });
  }
}

audio.addEventListener("ended", checkAudio);
document.getElementById("record-btn").addEventListener("click", voiceToggle);

const workletUrl = URL.createObjectURL(
  new Blob(
    [
      "(",
      function () {
        class WorkletProcessor extends AudioWorkletProcessor {
          constructor(options) {
            super();
            this.toggle = false;
            this.port.onmessage = this.onmessage.bind(this);
          }

          process(inputs, outputs) {
            if (this.toggle) {
              this.port.postMessage({
                event: "audio_data",
                data: inputs[0][0],
              });
            }

            return true;
          }

          onmessage(message) {
            const { event, data } = message.data;

            if (event === "toggle") {
              this.toggle = data;
            }
          }
        }

        registerProcessor("worklet-processor", WorkletProcessor);
      }.toString(),
      ")()",
    ],
    { type: "application/javascript" }
  )
);

async function init() {
  if (!navigator.mediaDevices) {
    console.log("getUserMedia not supported.");
    return;
  }

  const constraints = {
    audio: true,
    video: false,
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const microphone = audioContext.createMediaStreamSource(stream);

  await audioContext.audioWorklet.addModule(workletUrl);
  processor = new AudioWorkletNode(audioContext, "worklet-processor", {});
  processor.port.onmessage = (e) => {
    const { event, data } = e.data;

    if (event === "audio_data") {
      const buf = new Int16Array(data.length);
      let pcmSum = 0;
      for (let i = 0; i < data.length; i++) {
        const pcm = data[i];
        const pcm16 = Math.round(32767 * pcm);
        buf[i] = pcm16;
        pcmSum += Math.abs(pcm);
      }

      loudness = pcmSum / data.length;

      if (loudness < LOUDNESS_THRESHOLD) {
        const now = Date.now();
        if (!silenceStart) {
          silenceStart = Date.now();
        } else if (SILENCE_TIME_MS < now - silenceStart) {
          // 전송 안함
          // return;
        }
      } else if (silenceStart) {
        silenceStart = null;
      }

      if (socket) {
        socket.volatile.emit("audio_data", buf);
      }
    }
  };

  microphone.connect(processor);
  console.log("processor start");
}
init();

// OFF일 때 0을 계속 그림
setInterval(() => {
  if (!btn.classList.contains("on")) {
    s_series.append(Date.now(), 0);
    l_series.append(Date.now(), 0);
  } else {
    l_series.append(Date.now(), loudness);
    s_series.append(Date.now(), userScore);
  }
}, 100);
