@echo off


echo [Step 4] Installing required Python packages...
call .venv\Scripts\activate.bat
pip install --upgrade sentence-transformers llama-index-embeddings-huggingface

echo Starting phonex RAG server...

:: Tune Ollama for i7-1270P (16 threads, Intel Xe GPU)
set OLLAMA_NUM_THREAD=12
set OLLAMA_NUM_GPU=1
set OLLAMA_GPU_LAYERS=20
set OLLAMA_FLASH_ATTENTION=1

ollama run qwen2.5-coder:7b-instruct-q4_K_M "hello"
:: Start the Python server
cd /d D:\Gitrnd\phonex
python server.py

