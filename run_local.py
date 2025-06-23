#!/usr/bin/env python3
"""
Local development runner for Drishya
This script helps run the app locally with proper configuration
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_virtual_env():
    """Check if running in virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def main():
    print("🚀 Starting Drishya locally...\n")
    
    # Check if in virtual environment
    if not check_virtual_env():
        print("⚠️  Warning: Not running in a virtual environment")
        print("💡 Recommended: Create and activate a virtual environment first")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print()
    
    # Check if main file exists
    if not os.path.exists('sam-roboflow.py'):
        print("❌ sam-roboflow.py not found in current directory")
        print("💡 Make sure you're in the project root directory")
        return False
    
    try:
        print("📦 Installing/updating dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed")
        
        print("🌐 Starting Streamlit server...")
        print("📱 The app will open in your browser automatically")
        print("🔗 Local URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server\n")
        
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'sam-roboflow.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ])
        
        # Wait a moment then open browser
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:8501')
        except:
            pass  # Browser opening is optional
        
        # Wait for process to complete
        process.wait()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
