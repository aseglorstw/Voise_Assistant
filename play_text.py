import torch
import sounddevice as sd


def play(text):
    language = 'ru'
    speaker = 'baya_v2'
    sample_rate = 16000
    device = torch.device('cpu')
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                         model='silero_tts',
                                         language=language,
                                         speaker=speaker)
    model.to(device)  # gpu or cpu
    audio = model.apply_tts(texts=[text],
                            sample_rate=sample_rate)
    print(text)
    sd.play(audio[0], sample_rate)
    sd.wait()


play("привет")
