import torch
import sounddevice as sd

language = 'ru'
speaker = 'baya_v2'
sample_rate = 16000
device = torch.device('cpu')
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=speaker)
example_text = "Здарова, заебал!"
model.to(device)  # gpu or cpu
audio = model.apply_tts(texts=[example_text],
                        sample_rate=sample_rate)
print(example_text)
sd.play(audio[0], sample_rate)
sd.wait()



