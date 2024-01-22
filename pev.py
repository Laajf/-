import os
import sys
import subprocess
import platform

def install_msvc():
    try:
        import winreg
    except ImportError:
        print("Error: pywin32 module is not installed. Please run 'pip install pywin32'")
        sys.exit(1)

    msvc_version = "2019"  # Или "2017" в зависимости от вашей версии Visual C++ Build Tools

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\WOW6432Node\\Microsoft\\VisualStudio\\{msvc_version}.0")
    except FileNotFoundError:
        print(f"Error: Visual C++ {msvc_version} not found. Please install it manually.")
        sys.exit(1)

    vc_path = winreg.QueryValueEx(key, "InstallDir")[0]
    winreg.CloseKey(key)

    vcvars_path = os.path.join(vc_path, "VC\\Auxiliary\\Build\\vcvarsall.bat")

    subprocess.run([vcvars_path, "x86_amd64"], shell=True, check=True)

if __name__ == "__main__":
    if platform.system() == "Windows":
        install_msvc()
        print("Visual C++ Build Tools installed successfully.")
    else:
        print("Error: This script is intended for Windows only.")
