from potassium import Potassium, Request, Response, send_webhook
import whisper
import whisper.utils
import requests

audioFileName = 'input.mp3'

app = Potassium('whisper')

# @app.init runs at startup, and initializes the app's context
@app.init
def init():
    model = whisper.load_model("medium")

    context = {
        "model": model,
    }

    return context


@app.async_handler("/async")
def handler(context: dict, request: Request) -> Response:
    audio = request.json.get('audio')
    model = context.get('model')

    audioFile = downloadAudio(audio)
    result = model.transcribe(audioFile)
    print('Transcribe finished')

    writer = whisper.utils.get_writer('all', '.')
    writer(result, audioFile)

    webhook = request.json.get('webhook')
    send_webhook(url=webhook, json={"output": constructOutput()})

    return


def downloadAudio(url):
    print('Downloading audio from ', url)
    response = requests.get(url)
    print('Status code: ', response.status_code)
    open(audioFileName, 'wb').write(response.content)
    return audioFileName


def constructOutput():
    withoutExtension = audioFileName.removesuffix('.mp3')
    return {
        'text': open(f'{withoutExtension}.txt', encoding='utf-8').read(),
        'transcription': open(f'{withoutExtension}.vtt', encoding='utf-8').read(),
        'srt': open(f'{withoutExtension}.srt', encoding='utf-8').read(),
        'segments': open(f'{withoutExtension}.srt', encoding='utf-8').read()
    }

if __name__ == "__main__":
    app.serve()
