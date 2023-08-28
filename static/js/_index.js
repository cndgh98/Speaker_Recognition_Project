io({ transports: ["websocket"] }); //, upgrade: false
// const socket = io.connect("http://127.0.0.1:8080");
const socket = io.connect(null, { port: 8080, rememberTransport: false });
let processor = null;
let sid = null;
let loudness = 0;
let silenceStart = 0;

const LOUDNESS_THRESHOLD = 0.005;
const SILENCE_TIME_MS = 3000;

scoreEl = document.getElementById("score");
predictionEl = document.getElementById("prediction");
loudnessEl = document.getElementById("loudness");
messagesEl = document.getElementById("messages");

socket.on("connect", function () {
  init();
});

socket.on("connected", function ({ sid }) {
  console.log("sid:", sid);
});

socket.on("voice_recognition", function ({ score, prediction }) {
  console.log("voice_recognition:", score, prediction);
  scoreEl.textContent = score.toFixed(2);
  predictionEl.textContent = prediction ? "O" : "X";
});

socket.on("message", function (message) {
  messagesEl.value += message + "\n";
});

function voiceStart() {
  processor.port.postMessage({ event: "toggle", data: true });
}

function voiceStop() {
  processor.port.postMessage({ event: "toggle", data: false });
  socket.emit("audio_stop");
}

document.getElementById("start-btn").addEventListener("click", voiceStart);
document.getElementById("stop-btn").addEventListener("click", voiceStop);

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

      socket.emit("audio_data", buf);
    }
  };

  microphone.connect(processor);
  console.log("processor start");
}

setInterval(() => {
  loudnessEl.textContent = loudness.toFixed(4);
  if (loudness < LOUDNESS_THRESHOLD) {
    loudnessEl.style.color = "red";
  } else {
    loudnessEl.style.color = "green";
  }
}, 200);
