@echo off
cd /d "%~dp0"
echo === Creando entorno virtual en %~dp0venv ===
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
echo === Listo ===
pause
