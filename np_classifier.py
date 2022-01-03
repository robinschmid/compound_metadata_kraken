import time
import asyncio
import urllib
from aiohttp import ClientSession
from requests.exceptions import HTTPError

NP_CLASSIFIER_URL = "https://npclassifier.ucsd.edu/classify?smiles={}"

async def get(url, session):
    try:
        response = await session.request(method='GET', url=url)
        response.raise_for_status()
        # print(f"Response status ({url}): {response.status}")
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error ocurred: {err}")
    response_json = await response.json()
    return response_json


async def run_program(unique_canonical_smiles_dict, smiles, session):
    try:
        response = await get(np_class_url(smiles), session)
        unique_canonical_smiles_dict[smiles] = response
    except Exception as err:
        print(f"Exception occured: {err}")
        pass


async def get_all_np_class(unique_canonical_smiles_dict):
    async with ClientSession() as session:
        await asyncio.gather(*[run_program(unique_canonical_smiles_dict, smiles, session) for smiles in
                               unique_canonical_smiles_dict])


def np_class_url(smiles):
    return NP_CLASSIFIER_URL.format(urllib.parse.quote(smiles))

def get_json_dict(unique_canonical_smiles_dict):
    s = time.perf_counter()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_all_np_class(unique_canonical_smiles_dict))

    elapsed = time.perf_counter() - s

    # print(f"Completed {len(urls)} requests with {len(RESULTS)} results")
    print(elapsed)
    return unique_canonical_smiles_dict
