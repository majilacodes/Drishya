#!/usr/bin/env python3
"""
Deployment verification script for Drishya
Run this script to verify all dependencies and configurations are correct
"""

import sys
import importlib
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'torch',
        'torchvision', 
        'numpy',
        'cv2',
        'PIL',
        'segment_anything',
        'requests',
        'tqdm',
        'streamlit_drawable_canvas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_files():
    """Check if all required files exist"""
    required_files = [
        'sam-roboflow.py',
        'requirements.txt',
        'packages.txt',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def check_streamlit():
    """Check if Streamlit can run"""
    try:
        result = subprocess.run(['streamlit', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Streamlit - {version}")
            return True
        else:
            print("❌ Streamlit - Cannot run")
            return False
    except Exception as e:
        print(f"❌ Streamlit - Error: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 Verifying Drishya deployment readiness...\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Files", lambda: check_files()[0]),
        ("Dependencies", lambda: check_dependencies()[0]),
        ("Streamlit", check_streamlit)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}:")
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} - Error: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    
    if all_passed:
        print("🎉 All checks passed! Ready for deployment.")
        print("\n📝 Next steps:")
        print("1. Commit your changes: git add . && git commit -m 'Ready for deployment'")
        print("2. Push to GitHub: git push origin main")
        print("3. Deploy on Streamlit Cloud: https://share.streamlit.io")
    else:
        print("❌ Some checks failed. Please fix the issues above before deploying.")
        
        # Provide specific guidance
        deps_passed, missing_deps = check_dependencies()
        if not deps_passed:
            print(f"\n💡 To install missing dependencies:")
            print(f"   pip install {' '.join(missing_deps)}")
        
        files_passed, missing_files = check_files()
        if not files_passed:
            print(f"\n💡 Missing files: {', '.join(missing_files)}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
