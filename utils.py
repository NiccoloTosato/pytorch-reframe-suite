# SPDX-FileCopyrightText: 2026 Niccolo Tosato niccolo.tosato@yahoo.it
#
# SPDX-License-Identifier: MIT

import re
import urllib.request
import sys
def GetLatestPytorch(rocm_version, count=3):
    """Fetches the last N PyTorch versions for a given ROCm release."""
    index_url = f"https://download.pytorch.org/whl/rocm{rocm_version}/torch/"
    try:
        req = urllib.request.Request(index_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
        # This captures '2.10.0' but ONLY if '+rocm[version]-' immediately follows it.
        pattern = re.compile(r'torch-([\d\.]+)\+rocm[\d\.]+-')
        
        versions = sorted(list(set(pattern.findall(html))),
                          key=lambda x: [int(part) for part in x.split('.')])
        return versions[-count:] if versions else ["unknown"]
    except Exception as e:
        print(f"Warning: Failed to fetch versions. {e}")
        return ["unknown"]            

def GetCliVar(var_name, default_value):
    """
    Scrapes the raw command-line arguments (sys.argv) during ReFrame's 
    parsing phase to extract -S variables before tests are instantiated.
    WARNING THIS IS AN HACK !
    """
    prefix = f"{var_name}="
    for i, arg in enumerate(sys.argv):
        # Matches format: -S var_name=value
        if arg == '-S' and i + 1 < len(sys.argv) and sys.argv[i+1].startswith(prefix):
            return sys.argv[i+1].split('=', 1)[1].strip('"\'')
        # Matches format: -Svar_name=value
        elif arg.startswith('-S') and arg[2:].startswith(prefix):
            return arg.split('=', 1)[1].strip('"\'')
    return default_value
