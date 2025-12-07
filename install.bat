@echo off
SETLOCAL

:: -------------------------------
:: 1️⃣ Find Python 3.10
:: -------------------------------
for /f "tokens=*" %%i in ('py -0p ^| findstr "3.10"') do set PYTHON310=%%i
IF "%PYTHON310%"=="" (
    echo ============================================================
    echo Python 3.10 not found! Please install Python 3.10 first.
    echo Download: https://www.python.org/downloads/release/python-31011/
    echo ============================================================
    pause
    exit /b
)

echo Using Python: %PYTHON310%

:: -------------------------------
:: 2️⃣ Create virtual environment
:: -------------------------------
IF EXIST venv (
    echo Virtual environment 'venv' already exists. Skipping creation...
) ELSE (
    C:\Users\HP\AppData\Local\Programs\Python\Python310\python.exe -m venv venv
    echo Created virtual environment 'venv'.
)

:: -------------------------------
:: 3️⃣ Activate virtual environment
:: -------------------------------
call venv\Scripts\activate

:: -------------------------------
:: 4️⃣ Upgrade pip
:: -------------------------------
:: python -m pip install --upgrade pip

:: -------------------------------
:: 5️⃣ Install NumPy first (required by PyTorch)
:: -------------------------------
echo Installing stable NumPy...
pip install numpy==1.26.1

:: -------------------------------
:: 6️⃣ Install CUDA-enabled PyTorch (CUDA 12.4 for RTX 40xx)
:: -------------------------------
echo Installing PyTorch + CUDA 12.4...
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124  torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124

:: -------------------------------
:: 7️⃣ Install remaining requirements (excluding torch)
:: -------------------------------
echo Installing remaining packages...
pip install --no-deps -r requirements.txt

:: -------------------------------
:: 8️⃣ Test GPU availability
:: -------------------------------
echo ============================================================
echo Testing CUDA / GPU...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
echo ============================================================

:: -------------------------------
:: 9️⃣ Done
:: -------------------------------
echo Setup complete! Your virtual environment 'venv' is ready.
echo To activate, run: call venv\Scripts\activate
echo ============================================================
pause
ENDLOCAL
