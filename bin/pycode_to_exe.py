from PyInstaller.__main__ import run


if __name__ == '__main__':
    # opts = ['rw_test.py', '-F']
    # opts = ['test.py', '-D', "--add-data=D:/zc/htxt/bin/1.tiff;.", '--clean', '-y']
    # opts = ['test.py', '-D', "--add-data=D:/zc/htxt/bin/1.tiff;./img", '--clean', '-y']
    # opts = ['test.py', '-D', '--clean', '-y']
    opts = ['main.py', '-D','--clean', '-y',
            "--add-data=model;model",
            '--hidden-import=scipy._lib.messagestream',
            '--hidden-import=pywt._extensions._cwt',
            '--hidden-import=skimage.io._plugins',
            '--hidden-import=skimage.io._plugins.matplotlib_plugin',
            '--hidden-import=skimage.io._plugins.pil_plugin',
            '--hidden-import=skimage.io._plugins.tifffile_plugin'
            ]
    run(opts)

# pyinstaller test.py -D --add-data=D:\zc\htxt\bin\1.tiff;.\img --clean