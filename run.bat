@echo off
setlocal
cd /d "%~dp0"
title Femto Bolt Pose

if exist ".venv\setup_done.txt" goto run

echo ============================================================
echo  First run: setting up (one time only, takes a few minutes)
echo ============================================================
call :findpy
if not defined PYEXE (
    echo Python 3.12 not found - installing it with winget...
    winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
    call :findpy
)
if not defined PYEXE (
    echo.
    echo Could not find or install Python 3.12.
    echo Install it from https://www.python.org/downloads/release/python-3120/ then double-click this file again.
    goto fail
)
echo Using %PYEXE%
"%PYEXE%" -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
echo ok> ".venv\setup_done.txt"

:run
echo Starting... (press q or Esc in the video window to quit)
".venv\Scripts\python.exe" femto_pose.py %*
if errorlevel 1 goto fail
exit /b 0

:findpy
set "PYEXE="
for /f "delims=" %%i in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%ProgramFiles%\Python312\python.exe" set "PYEXE=%ProgramFiles%\Python312\python.exe"
exit /b 0

:fail
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
