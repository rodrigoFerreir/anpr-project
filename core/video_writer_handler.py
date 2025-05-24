import os
import cv2
import logging

logger = logging.getLogger(__name__)


class VideoWriterHandler:
    def __init__(self):
        """
        Inicializa o manipulador de escrita de vídeo.

        :param output_path: Caminho para o arquivo de saída (ex: 'output.mp4')
        :param fps: Frames por segundo do vídeo de saída
        :param frame_size: Tamanho dos frames (largura, altura)
        :param codec: Codec a ser utilizado (padrão: 'mp4v')
        :param is_color: Indica se o vídeo é colorido
        """
        self.output_path = None
        self.fps = None
        self.frame_size = None
        self.codec = "mp4v"
        self.is_color = True
        self.writer = None

    def _initialize_writer(
        self,
        output_path,
        fps,
        frame_size,
        codec="mp4v",
        is_color=True,
    ):
        """
        Inicializa o objeto VideoWriter com os parâmetros fornecidos.
        """
        # Garante que o diretório de saída existe
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.is_color = is_color

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Define o codec
        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        # Inicializa o VideoWriter
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size,
            self.is_color,
        )

        if not self.writer.isOpened():
            logger.error(
                f"Não foi possível abrir o arquivo de vídeo para escrita: {self.output_path}"
            )
            raise IOError(f"Falha ao abrir o arquivo de vídeo: {self.output_path}")
        else:
            logger.info(f"Arquivo de vídeo iniciado para escrita: {self.output_path}")

    def write_frame(self, frame):
        """
        Escreve um frame no arquivo de vídeo.

        :param frame: Frame a ser escrito
        """
        if self.writer:
            self.writer.write(frame)
        else:
            logger.warning("Tentativa de escrever frame sem inicializar o VideoWriter.")

    def release(self):
        """
        Libera os recursos associados ao VideoWriter.
        """
        if self.writer:
            self.writer.release()
            logger.info(f"Arquivo de vídeo finalizado: {self.output_path}")
