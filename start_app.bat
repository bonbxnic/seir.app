@echo off
TITLE SEIR Epidemic Simulator
ECHO Starting SEIR Epidemic Analysis Tool...
ECHO.
ECHO Please wait while the application loads...
ECHO.

REM Check if virtual environment exists
IF EXIST ".venv\Scripts\python.exe" (
    REM Activate and run
    CALL .venv\Scripts\activate.bat
    streamlit run app.py
) ELSE (
    ECHO Virtual environment not found!
    ECHO Please ensure you have set up the project correctly.
    ECHO Attempting to run with system python...
    python -m streamlit run app.py
)

PAUSE