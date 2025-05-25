import os
import cv2
import time
import queue
import logging
import threading


logger = logging.getLogger(__name__)


class VideoStreamProcessor:
    def __init__(
        self,
        stream_url: str,
        inference: object = None,
        writer: object = None,
        max_retries: int = 5,
        retry_delay: int = 5,
        save_result: bool = True,
        output_dir: str = "result",
    ):
        self.output_dir = output_dir
        self.stream_url = stream_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.save_result = save_result

        # Dependências e módulos de apoio.
        self.inference = inference
        self.writer = writer

        # Propriedades do stream e controle de estado.
        self.capture: cv2.VideoCapture = None
        self.frame_queue: queue.Queue = queue.Queue()  # Fila sem tamanho máximo
        self.running: bool = True
        self.frame_count: int = 0
        self.total_frames: int = None
        self.frame_size: tuple = (0, 0)
        self.fps: float = 0.0

        # Threads para captura e processamento.
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            name="CaptureThread",
        )
        self.processor_thread = threading.Thread(
            target=self._process_frames,
            name="ProcessThread",
        )
        self.processor_thread_1 = threading.Thread(
            target=self._process_frames,
            name="ProcessThread-1",
        )
        self.processor_thread_2 = threading.Thread(
            target=self._process_frames,
            name="ProcessThread-2",
        )
        self.processor_thread_3 = threading.Thread(
            target=self._process_frames,
            name="ProcessThread-3",
        )

    def _connect_stream(self) -> bool:
        """
        Tenta estabelecer a conexão com o stream, configurando as propriedades necessárias.

        - Abre o stream e, se bem-sucedido, coleta: tamanho do frame, FPS e total de frames (se finito);
        - Caso esteja habilitado o salvamento, inicializa o VideoWriter.
        - Se exceder o número máximo de tentativas, define 'running' como False.

        Retorna:
          True  -> Conexão estabelecida com sucesso.
          False -> Falha na conexão após as tentativas.
        """
        retries = 0
        while retries < self.max_retries and self.running:
            self.capture = cv2.VideoCapture(self.stream_url)
            time.sleep(1)
            if self.capture and self.capture.isOpened():
                logger.info("Conectado ao stream.")
                self._initialize_stream_properties()
                if self.save_result:
                    self._initialize_video_writer()
                return True

            retries += 1
            logger.warning(f"Falha na conexão. Tentativa {retries}/{self.max_retries}.")
            time.sleep(self.retry_delay)

        logger.error("Reconexão falhou após múltiplas tentativas. Encerrando captura.")
        self.running = False
        return False

    def _initialize_stream_properties(self) -> None:
        """
        Coleta e inicializa as propriedades essenciais do stream:
          - Tamanho dos frames.
          - Taxa de frames por segundo (FPS).
          - Total de frames, caso o stream seja finito.
        """
        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (frame_width, frame_height)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        total_frames_candidate = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_candidate > 0:
            self.total_frames = total_frames_candidate
            logger.info(f"Vídeo finito detectado: {self.total_frames} frames.")
        else:
            self.total_frames = None
            logger.info("Stream infinito ou total de frames não definido.")

    def _initialize_video_writer(self) -> None:
        """
        Inicializa o VideoWriter para salvar o resultado, definindo o caminho com base na URL.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        output_filename = f"{os.path.basename(self.stream_url)}"
        output_path = os.path.join(self.output_dir, output_filename)
        logger.info("Inicializando o VideoWriter para salvar o resultado...")
        self.writer._initialize_writer(output_path, self.fps, self.frame_size)

    def verify_stream_finished(self) -> bool:
        if self.total_frames and self.frame_count >= self.total_frames:
            logger.info("Encerrando captura.")
            self.running = False
            return True
        return False

    def _capture_frames(self) -> None:
        """
        Loop de captura de frames com reconexões automáticas.

        Estrutura:
          1. Loop externo: enquanto 'running' for True, tenta conectar ao stream.
          2. Loop interno: captura frames enquanto a conexão estiver ativa.
             - Se a leitura falhar e o vídeo for finito (frame_count >= total_frames), encerra a captura.
             - Em caso de falha, libera a captura e sai do loop interno para tentar reconectar.
          3. Aguarda 'retry_delay' entre as tentativas de reconexão.
        """
        while self.running:
            if not self._connect_stream():
                break  # Não foi possível conectar; encerra a captura.

            # Loop interno: captura frames enquanto a conexão estiver OK.
            while self.running:
                ret, frame = self.capture.read()
                if not ret:
                    # Se for um vídeo finito e já tiver lido todos os frames, encerra sem reconectar.
                    if self.verify_stream_finished():
                        self.capture.release()
                        break

                    logger.warning("Falha na leitura do frame. Tentando reconectar...")
                    self.capture.release()
                    break  # Sai do loop interno para tentar reconectar.

                self.frame_queue.put(frame)
                self.frame_count += 1

                # Se o vídeo for finito e atingiu o total de frames, interrompe a captura.
                if self.verify_stream_finished():
                    self.capture.release()
                    break

            # Se 'running' ainda estiver True, aguarda antes de tentar reconectar.
            if self.running:
                logger.info("Tentando reconectar ao stream...")
                time.sleep(self.retry_delay)

        logger.info("Thread de captura encerrada.")

    def _process_frames(self) -> None:
        """
        Processa frames enfileirados até que a captura seja interrompida e a fila fique vazia.

        Para cada frame, é realizado:
          - Processamento (inferência) através do módulo 'inference'.
          - Salvamento do frame processado, caso 'save_result' esteja ativado.
        """
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get()
                processed_frame = self.inference.inference(frame)
                if self.save_result:
                    self.writer.write_frame(processed_frame)
                # Eventualmente, aqui você pode adicionar exibição ou outro tratamento.
                print(self.frame_queue.qsize())
            except queue.Empty:
                continue

        logger.info("Thread de processamento encerrada.")

        if self.save_result:
            logger.info(
                f"Salvando resultado {os.path.join(self.output_dir, self.stream_url)}."
            )
            self.writer.release()

    def start(self) -> None:
        """
        Inicia as threads de captura e processamento, permitindo que a operação ocorra de forma paralela.
        """
        logger.info("Iniciando o processamento do stream...")

        if self.inference is None:
            raise ValueError("O módulo de inferência não foi inicializado.")

        if self.writer is None and self.save_result:
            raise ValueError("O módulo de escrita de vídeo não foi inicializado.")

        self.capture_thread.start()
        self.processor_thread.start()
        self.processor_thread_1.start()
        self.processor_thread_2.start()
        self.processor_thread_3.start()

    def stop(self) -> None:
        """
        Finaliza o processamento:
          - Define 'running' como False para sinalizar o encerramento dos loops.
          - Libera recursos associados ao stream e ao VideoWriter.
          - Aguarda o término das threads.
        """
        self.running = False
        if self.capture and self.capture.isOpened():
            self.capture.release()

        self.capture_thread.join()
        self.processor_thread.join()
        self.processor_thread_1.join()
        self.processor_thread_2.join()
        self.processor_thread_3.join()
        logger.info("Processamento encerrado com sucesso.")
