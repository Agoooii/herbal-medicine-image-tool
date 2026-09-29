@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -B "%~dp0程序文件\process.py"
if errorlevel 1 pause
