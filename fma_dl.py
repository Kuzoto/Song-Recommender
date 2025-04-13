import requests
import zipfile
import io

r = requests.get("https://os.unil.cloud.switch.ch/fma/fma_small.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./train_db")

r = requests.get("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./train_db")