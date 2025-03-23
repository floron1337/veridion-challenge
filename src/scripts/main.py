import pandas as pd
import utils
import constants

df = pd.read_parquet("../logos.snappy.parquet", engine="pyarrow")

for row in df.itertuples():
    domain = row[1]

    logo_file = utils.download_logo(domain, dest=constants.LOGOS_FOLDER)
    if logo_file is None:
        print(f"Error downloading logo for domain {domain}")
        continue

    print(domain)

    utils.delete_file(logo_file)