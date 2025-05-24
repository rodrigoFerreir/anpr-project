# Python OCR Detection

![License Plate Detection](https://img.shields.io/badge/OCR-EasyOCR-blue)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-green)
![Python 3.11](https://img.shields.io/badge/Python-3.11+-yellow)

## Visão Geral

Este projeto realiza **detecção automática de placas de veículos** em vídeos ou streams, utilizando o modelo YOLOv8 para detecção e EasyOCR para reconhecimento óptico de caracteres (OCR). O pipeline é totalmente automatizado: baixa o vídeo e o modelo, processa o stream, realiza inferência e salva o resultado anotado.

---

## Funcionalidades

- **Detecção de placas** com YOLOv8 ([Ultralytics](https://github.com/ultralytics/ultralytics))
- **Reconhecimento de caracteres** com EasyOCR
- Processamento de vídeo em tempo real ou arquivos locais
- Escrita de vídeo processado com anotações
- Reconexão automática em caso de falha no stream
- Modular e extensível (adapters para modelos e OCR)

---

## Instalação

1. **Clone o repositório:**
   ```sh
    git clone https://github.com/seu-usuario/python-ocr-detection.git
    cd python-ocr-detection
   ```

2. **Crie um ambiente virtual e ative-o:**
    ```
        python3.11 -m venv .venv
        source .venv/bin/activate
    ```

1. **Instale as dependências:**
   ``` pip install -r requirements.txt ```
    **ou, se preferir, usando o pyproject.toml:**
    ``` pip install . ```

