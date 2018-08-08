from PyInstaller.__main__ import run


if __name__ == '__main__':
    # opts = ['rw_test.py', '-F']
    # opts = ['test.py', '-D', "--add-data=D:/zc/htxt/bin/1.tiff;.", '--clean', '-y']
    # opts = ['test.py', '-D', "--add-data=D:/zc/htxt/bin/1.tiff;./img", '--clean', '-y']
    opts = ['main.py', '-D', "--add-data=model;model",
            '--clean', '-y']
    run(opts)

# pyinstaller test.py -D --add-data=D:\zc\htxt\bin\1.tiff;.\img --clean