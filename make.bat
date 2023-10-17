CALL venv\Scripts\activate
pyinstaller --log-level=DEBUG --add-binary "./dlls/*;./" --add-binary "./venv/Lib/site-packages/libtiepie*;./" --add-binary "./settings/*;./settings" --add-binary "./lockin.ui;./" --add-binary "./interface.ui;./" --noconfirm --hiddenimport libtiepie "app.py"
pause