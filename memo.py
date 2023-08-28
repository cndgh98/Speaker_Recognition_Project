
# self.thread = threading.Thread(target=self.stt_worker)
# self.thread.daemon = True
# self.thread.start()
# self.thread.join()


# @socketio.on("create_profile")
# def on_select_profile(profile_name):
#     sid = request.sid
#     for profile in profiles:
#         if profile["name"] == profile_name:
#             emit("create_profile", {"profile": None}, to=sid)
#             break
#     else:
#         connections[sid].profile = profile_name
#         emit("create_profile", {
#              "profile": profile_name, "recording": 0}, to=sid)


# @socketio.on("select_profile")
# def on_select_profile(profile_name):
#     sid = request.sid
#     for profile in profiles:
#         if profile["name"] == profile_name:
#             connections[sid].profile = profile
#             connections[sid].emb_tensor = torch.Tensor(profile["emb"])
#             emit("select_profile", {"profile": profile_name}, to=sid)
#             break
#     else:
#         emit("select_profile", {"profile": None}, to=sid)


# @socketio.on("audio_data")
# def audio_data(data):
#     sid = request.sid
#     connection = connections[sid]

#     if connection["profile"]:
#         unpack_data = struct.unpack('128h', data)
#         wave = [pcm / 32767 for pcm in unpack_data]
#         connection["buf"] += wave
#         lock.acquire()
#         connection["stt_buf"] += data
#         lock.release()

#         loudness_sample = connection["buf"][-int(
#             SAMPLE_RATE * LOUDNESS_SAMPLE_TIME):]
#         loudness = mean([abs(x) for x in loudness_sample])
#         if loudness < LOUDNESS_THRESHOLD:
#             connection["buf"] = connection["buf"][:-len(loudness_sample)]

#         if BUF_SIZE < len(connection["buf"]):
#             connection["buf"] = connection["buf"][-BUF_SIZE:]

#         if BUF_SIZE <= len(connection["buf"]):
#             if type(connection["profile"]) == str:
#                 # 프로필 생성
#                 emb = get_emb_by_pcm(connection["buf"])
#                 emb_data = emb.data.cpu().numpy()[0].tolist()
#                 profile = {
#                     "name": connection["profile"],
#                     "emb": emb_data
#                 }
#                 profiles.append(profile)
#                 connection["emb_tensor"] = emb
#                 connection["profile"] = profile

#                 with open("profiles.json", "w") as f:
#                     json.dump(profiles, f)

#                 emit("create_profile", {
#                     "profile": profile["name"], "recording": 1}, to=sid)
#             elif connection["last_tick"] + TICK_TIME <= time.time():
#                 profile_emb = connection["emb_tensor"]
#                 emb = get_emb_by_pcm(connection["buf"])
#                 score = similarity(profile_emb, emb).item()
#                 prediction = THRESHOLD < score
#                 emit("voice_recognition", {
#                     "score": score, "prediction": prediction}, to=sid)

#                 connection["last_tick"] = time.time()
