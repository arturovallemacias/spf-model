@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python panel.py
pause
