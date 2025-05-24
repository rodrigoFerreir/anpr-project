import os
import time
import logging
import easyocr

from core.inference import Inference
from core.adapters.yolo_adapter import YoloAdapter
from core.adapters.easyocr_adapter import EasyOCRAdapter
from core.video_writer_handler import VideoWriterHandler
from core.video_stream_processor import VideoStreamProcessor

from ultralytics.utils.downloads import safe_download


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BASE_DOWNLOAD = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"


def main():
    # Defina a URL do stream conforme o seu ambiente
    stream_url = "anpr-demo-video.mp4"
    model_name = "anpr-demo-model.pt"

    if not os.path.exists(stream_url):
        logging.info(f"Baixando video {stream_url}")
        safe_download(BASE_DOWNLOAD + stream_url)

    if not os.path.exists(model_name):
        logging.info(f"Baixando o modelo {model_name}")
        safe_download(BASE_DOWNLOAD + model_name)

    reader = EasyOCRAdapter(["en"])
    model_anpr = YoloAdapter(model_name)
    inference = Inference(model_anpr, reader)
    writer = VideoWriterHandler()

    processor = VideoStreamProcessor(
        stream_url,
        inference=inference,
        writer=writer,
        save_result=True,
    )

    try:
        processor.start()
        # Loop principal para manter o programa ativo
        while processor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Encerramento solicitado pelo usuário.")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()
