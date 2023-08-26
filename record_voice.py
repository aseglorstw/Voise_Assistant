import sounddevice as sd
import queue
import json
import Words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import vosk

q = queue.Queue()
language = 'ru'
speaker = 'baya_v2'
sample_rate1 = 16000
device1 = torch.device('cpu')
model1, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=speaker)
model1.to(device1)


def callback111(indata, frames, time, status):
    q.put(bytes(indata))


def main():
    model = vosk.Model('model_small')
    device = sd.default.device
    samplerate = int(sd.query_devices(device[0], 'input')['default_samplerate'])
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(Words.data_set.keys()))

    clf = LogisticRegression()
    clf.fit(vectors, list(Words.data_set.values()))

    del Words.data_set

    with sd.RawInputStream(samplerate=samplerate, blocksize=32000, device=device[0], dtype='int16', channels=1,
                           callback=callback111):
        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                play(json.loads(rec.Result())['text'], vectorizer, clf)


def play(text, vectorizer, clf):
    trg = Words.TRIGGERS.intersection(text.split())
    if not trg:
        return
    print("Yes")
    text.replace(list(trg)[0], '')
    text_vector = vectorizer.transform([text]).toarray()[0]
    answer = clf.predict([text_vector])[0]
    audio = model1.apply_tts(texts=[answer], sample_rate=sample_rate1)
    sd.play(audio[0], sample_rate1)
    sd.wait()


if __name__ == '__main__':
    main()
