@echo off

where python >nul 2>nul
if not %errorlevel%==0 (
    echo Python executable not found!
    exit /b 1
)

python %~dp0\extracter.py
pause
