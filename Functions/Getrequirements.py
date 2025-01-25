
def getrequirements(url):

    import requests
    import importlib
    import subprocess
    import sys
    import tempfile

    rspns = requests.get(url)
    rspns.raise_for_status()
    requ =  rspns.text
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(requ)
        tmpp = tmp.name
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", tmpp])
    imported_modules = {}
    for line in requ.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        package_name = (line
                        .split('==')[0])
        try:
            imported_modules[package_name] = importlib.import_module(package_name)
        except ImportError:
            print(f"Warning: Unable to import '{package_name}'")
    return imported_modules

#getrequirements('https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/DSWBranch/requirementsTEST.txt')

#ok so funny thing... it does import all things, it just says unable to import X because libraries have names that dont match their import names. scikit-learn -> sklearn, pillow - PIL ect
