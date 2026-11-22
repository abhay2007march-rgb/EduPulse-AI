@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run start.bat first to create the virtual environment.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python generate_data.py
python train_model.py
pause
