@echo off
setlocal
cd /d "%~dp0"
rem Registers the "Imagineers" virtual camera (Unity Capture DirectShow driver, in .\virtualcam).
rem Needs admin once per computer; asks for it with a Windows prompt if not already admin.
rem Usage: install_virtualcam.bat [name]      (default name: Imagineers)
set "VCNAME=%~1"
if "%VCNAME%"=="" set "VCNAME=Imagineers"

if not exist "%~dp0virtualcam\UnityCaptureFilter64.dll" (
    echo Missing virtualcam\UnityCaptureFilter64.dll - copy the whole femto-pose folder, including "virtualcam".
    exit /b 1
)

net session >nul 2>&1
if not errorlevel 1 goto register

echo Asking Windows for admin rights to install the "%VCNAME%" camera - click Yes...
set "VCBAT=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Start-Process -FilePath $env:ComSpec -ArgumentList ('/c \"\"' + $env:VCBAT + '\" \"' + $env:VCNAME + '\"\"') -Verb RunAs -Wait -ErrorAction Stop } catch { exit 1 }"
if errorlevel 1 (
    echo The admin prompt was declined - the camera was not installed.
    exit /b 1
)
exit /b 0

:register
echo Registering virtual camera "%VCNAME%"...
regsvr32 /s "%~dp0virtualcam\UnityCaptureFilter32.dll" "/i:UnityCaptureName=%VCNAME%"
regsvr32 /s "%~dp0virtualcam\UnityCaptureFilter64.dll" "/i:UnityCaptureName=%VCNAME%"
if errorlevel 1 (
    echo Registration failed.
    timeout /t 5 >nul
    exit /b 1
)
echo Done: "%VCNAME%" is now a camera. Restart any app that was already open to see it.
timeout /t 2 >nul
exit /b 0
