#!v:\myproject\env\scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'htmlmin==0.1.12','console_scripts','htmlmin'
__requires__ = 'htmlmin==0.1.12'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('htmlmin==0.1.12', 'console_scripts', 'htmlmin')()
    )
