REM run the SimSpark simulator without internal monitor and logging on

set SPARK_DIR=%CD%\simspark
set PATH=%CD%\simspark\lib\simspark;
set PATH=%CD%\simspark\lib\thirdparty;%PATH%
set PATH=%CD%\simspark\lib\rcssserver3d;%PATH%

cd simspark\bin
rcssserver3d.exe --script-path rcssserverspl_no_monitor.rb
if %ERRORLEVEL% GEQ 1 pause