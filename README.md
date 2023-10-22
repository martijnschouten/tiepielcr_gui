# TiePieLCR gui
This project provides a graphical interface for the tiepieLCR. The hardware of the tiepieLCR consists of a TiePie Engineering HS5 and a custom front end which can be downloaded from [here](https://github.com/martijnschouten/tiepielcr-hardware)

![GUI](https://github.com/martijnschouten/tiepielcr_gui/assets/6079002/29a33b71-5dbc-4170-9a40-ec1b3164a94f)

A precompiled version of the code in this repository can be downloaded from: https://martijnschouten.stackstorage.nl/s/dTPoGPGhHa92hxrv

Note that in order to use leverage the GPU for the demodulation cuda toolkit 11.3 should downloaded and installed. It can be downloaded from: https://developer.nvidia.com/cuda-downloads

# Instructions
In order run the python code:

windows:
1. download and install the tiepie HS5 usb driver from https://www.tiepie.com/en/usb-oscilloscope/handyscope-hs5#downloads
2. install cuda toolkit 11.3 from https://developer.nvidia.com/cuda-downloads using typical instalation settings
3. install python 3.9.9
4. pip install numpy-1.21.5+mkl-cp39-cp39-win_amd64.whl
5. pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
6. pip install python-libtiepie pyyaml pyqt5 pyqtgraph scipy matplotlib pyinstaller sphinx sphinx-rtd-theme
7. in \venv\Lib\site-packages\libtiepie open devicelistitem.py replace "return array('L', values) with "return values

Ubuntu:
1. install sdk, by unzipping and running install script: https://www.tiepie.com/en/libtiepie-sdk/linux
2. download github desktop: sudo wget https://github.com/shiftkey/desktop/releases/download/release-2.9.3-linux3/GitHubDesktop-linux-2.9.3-linux3.deb
3. install github desktop by double cliking on the deb
4. install nvidia toolkit 11.3 using https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu
