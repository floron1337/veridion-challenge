import constants
import requests
import os


# Obtine logo-ul unei pagini folosind un API si il salveaza sub forma de PNG
# la locatia [dest] se va stoca logo-ul sub forma [domain].PNG
def download_logo(domain, dest='.'):
    response = requests.get(url=f'{constants.CLEARBIT_LOGO_API_URL}/{domain}')
    if response.status_code == 200:
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, f'{domain}.png'), "wb") as file:
            file.write(response.content)
            return f'{dest}/{domain}.png'
    else:
        return None


# Sterge un fisier specificat de parametru
def delete_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
