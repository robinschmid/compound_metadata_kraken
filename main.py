from enum import Enum
from functools import partial
import pandas as pd
import requests
import urllib.parse
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils

from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit.Chem.rdMolDescriptors as Desc

import json
import math
import pprint

import np_classifier
# get specific logger
import logging
import logging.config
logging.config.fileConfig(fname='logger.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

halogens = [9, 17, 35, 53]


def count_element(mol, number):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == number)


def count_H(mol):
    return count_element(mol, 1)


def count_C(mol):
    return count_element(mol, 6)


def count_N(mol):
    return count_element(mol, 7)


def count_O(mol):
    return count_element(mol, 8)


def count_P(mol):
    return count_element(mol, 15)


def count_S(mol):
    return count_element(mol, 16)


def count_halogens(mol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in halogens)


def num_o_atoms(df):
    return df[Columns.rdkit_mol.name].apply(count_O)


def num_n_atoms(df):
    return df[Columns.rdkit_mol.name].apply(count_N)


def num_c_atoms(df):
    return df[Columns.rdkit_mol.name].apply(count_C)


def num_h_atoms(df):
    return df[Columns.rdkit_mol.name].apply(count_H)


def num_s_atoms(df):
    return df[Columns.rdkit_mol.name].apply(count_S)


def num_p_atoms(df):
    return df[Columns.rdkit_mol.name].apply(count_P)


def num_halogen_atoms(df):
    return df[Columns.rdkit_mol.name].apply(count_halogens)


def mol_formula(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: None if mol is None else Desc.CalcMolFormula(mol))


def exact_mass(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: round(Descriptors.ExactMolWt(mol), 5))


def mass_defect(df):
    return df[Columns.exact_mass.name].apply(lambda exact_mass: round(exact_mass - math.floor(exact_mass), 5))


def mol_weight(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: round(Descriptors.MolWt(mol), 5))


def NumValenceElectrons(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Descriptors.NumValenceElectrons(mol))


def calc_hbd_donor(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Desc.CalcNumHBD(mol))


def calc_hba_acceptor(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Desc.CalcNumHBA(mol))


def calc_rotatable_bonds(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Desc.CalcNumRotatableBonds(mol))


def num_hetero_atoms(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Desc.CalcNumHeteroatoms(mol))


def num_aromatic_rings(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Desc.CalcNumAromaticRings(mol))


def canonical_smiles(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Chem.MolToSmiles(mol, True))


def inchi(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Chem.MolToInchi(mol))


def inchi_key(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Chem.MolToInchiKey(mol))


def smarts(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Chem.MolToSmarts(mol))


def rdkit_mol(df):
    if Columns.canonical_smiles.name in df.columns:
        return df[Columns.canonical_smiles.name].apply(lambda smiles: Chem.MolFromSmiles(smiles))
    elif Columns.inchi.name in df.columns:
        return df[Columns.inchi.name].apply(lambda inchi: Chem.MolFromInchi(inchi))
    elif Columns.smiles.name in df.columns:
        return df[Columns.smiles.name].apply(lambda smiles: Chem.MolFromSmiles(smiles))
    else:
        raise AttributeError("Data frame with inchi or smiles column needed")


def get_original_structures(df):
    if Columns.canonical_smiles.name in df.columns:
        return df[Columns.canonical_smiles.name]
    elif Columns.inchi.name in df.columns:
        return df[Columns.inchi.name]
    elif Columns.smiles.name in df.columns:
        return df[Columns.smiles.name]
    else:
        raise AttributeError("Data frame with inchi or smiles column needed")


class Columns(Enum):
    rdkit_mol = partial(rdkit_mol)
    formula = partial(mol_formula)
    exact_mass = partial(exact_mass)
    mass_defect = partial(mass_defect)
    mw = partial(mol_weight)
    hba = partial(calc_hba_acceptor)
    hbd = partial(calc_hbd_donor)
    num_rot_bonds = partial(calc_rotatable_bonds)
    hetero_atoms = partial(num_hetero_atoms)
    h_atoms = partial(num_h_atoms)
    c_atoms = partial(num_c_atoms)
    n_atoms = partial(num_n_atoms)
    o_atoms = partial(num_o_atoms)
    p_atoms = partial(num_p_atoms)
    s_atoms = partial(num_s_atoms)
    halogen_atoms = partial(num_halogen_atoms)
    aromatic_rings = partial(num_aromatic_rings)
    valenz = partial(NumValenceElectrons)
    smiles = partial(canonical_smiles)
    canonical_smiles = partial(canonical_smiles)
    inchi = partial(inchi)
    inchi_key = partial(inchi_key)
    smarts = partial(smarts)

    def create_col(self, df):
        return self.__call__(df)

    def __str__(self):
        return self.namedef

    def __call__(self, *args):  # make it callable
        return self.value(*args)


NP_CLASSIFIER_URL = "https://npclassifier.ucsd.edu/classify?smiles={}"
CLASSYFIRE_URL = "https://gnps-structure.ucsd.edu/classyfire?smiles={}"
CLASSYFIRE_SUFFIX = "_classyfire"
NP_CLASSIFIER_SUFFIX = "_np_classifier"


# exact mass
# mass defect
# IDS ChEBI, PubChem, ChemSpider
# Formula
# DBE
# xlogp
# flatness
# H donor / acceptor
# therapeutic
# violin plots
# NPAtlas:
# ORIGIN ORGANISM TYPE
def search_chembl():
    # chembl
    molecule = new_client.molecule
    molecule.set_format('json')
    # aspirin = molecule.search('aspirin')
    aspirin = molecule.search('CC(=O)OCC(CCC=C(C)C)=CCCC(CO)=CCCC(C)=CCO')

    for r in aspirin:
        pref_name = r['pref_name']
        if pref_name is not None:
            print(pref_name)


def main():
    original_df = pd.read_csv("data/smiles.tsv", sep="\t")

    # create mol column and filter rows - missing mol means unparsable smiles or inchi
    try:
        original_df[Columns.rdkit_mol.name] = Columns.rdkit_mol.create_col(original_df)
        filtered_df = original_df[original_df[Columns.rdkit_mol.name].astype(bool)]
    except Exception as e:
        logger.error("Error while parsing molecular structures", e)
        exit(1)

    unparsable_rows = len(original_df) - len(filtered_df)
    if unparsable_rows>0:
        unparsed_df = original_df[original_df[Columns.rdkit_mol.name].astype(bool)==False]
        unparsed_structures = get_original_structures(unparsed_df)
        logger.info("n=%d rows (structures) were not parsed: %s", unparsable_rows, "; ".join(unparsed_structures))
    else:
        logger.info("All row structures were parsed")

    # add new columns for chemical properties
    for col in Columns:
        if col.name not in filtered_df:
            filtered_df[col.name] = col.create_col(filtered_df)

    # read classes from gnps APIs
    filtered_df = np_class(filtered_df)
    # filtered_df = classyfire(filtered_df)

    # read data bases
    # search_chembl(filtered_df)

    filtered_df.drop(columns=[Columns.rdkit_mol.name], axis=1, inplace=True)
    filtered_df.to_csv("results/converted.tsv", sep='\t', encoding='utf-8', index=False)

    exit(0)


def classyfire_url(smiles):
    return CLASSYFIRE_URL.format(urllib.parse.quote(smiles))


def np_class_url(smiles):
    return NP_CLASSIFIER_URL.format(urllib.parse.quote(smiles))


def get_json_response(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return json.loads(response.text)
    except requests.exceptions.HTTPError as errh:
        print("Http Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Other error:", err)
    # on error return None
    return None


def get_unique_canocical_smiles_dict(df):
    """
    Dict with unique canonical smiles as keys
    :param df: input data frame with Columns.canonical_smiles column
    :return: dict(canonical_smiles, None)
    """
    return dict.fromkeys(df[Columns.canonical_smiles.name])


def np_class(df):
    unique_smiles_dict = get_unique_canocical_smiles_dict(df)
    for smiles in unique_smiles_dict:
        unique_smiles_dict[smiles] = get_json_response(np_class_url(smiles))

    # temp column with json results
    result_column = df[Columns.canonical_smiles.name].apply(lambda smiles: unique_smiles_dict[smiles])
    # extract and join values from json array - only isglycoside is already a value
    df2 = pd.DataFrame()
    df2["class_results"] = result_column.apply(lambda json: join(json, "class_results"))
    df2["superclass_results"] = result_column.apply(lambda json: join(json, "superclass_results"))
    df2["pathway_results"] = result_column.apply(lambda json: join(json, "pathway_results"))
    df2["is_glycoside"] = result_column.apply(lambda json: json["isglycoside"] if json is not None else None)
    df2 = df2.add_suffix(NP_CLASSIFIER_SUFFIX)
    return df.join(df2)  # add to original df


# def np_class(df, unique_smiles_dict):
#     # Query NP classifier on GNPS
#     rs = (grequests.get(np_class_url(smiles)) for smiles in df[Columns.canonical_smiles.name])
#
#     responses = grequests.map(rs)
#     # convert json response to columns, add suffix, add to data frame
#     contents = [handle_response_to_json(response) for response in responses]
#     df2 = pd.DataFrame(contents)
#     # extract and join values from json array - only isglycoside is already a value
#     df2["class_results"] = df2["class_results"].apply(join)
#     df2["superclass_results"] = df2["superclass_results"].apply(join)
#     df2["pathway_results"] = df2["pathway_results"].apply(join)
#     df2 = df2.add_suffix(NP_CLASSIFIER_SUFFIX)
#     return df.join(df2)  # add to original df


def classyfire(original_df, unique_smiles):
    # Query classyfire on GNPS
    rs = (grequests.get(classyfire_url(smiles)) for smiles in original_df[Columns.canonical_smiles.name])
    responses = grequests.map(rs)
    # convert json response to columns, add suffix, add to data frame
    contents = [None if response is None or response.text is None else json.loads(response.text) for response in \
                responses]
    json_df = pd.DataFrame(contents)

    # extract information
    classy_df = pd.DataFrame()
    classy_df["kingdom"] = json_df["kingdom"].apply(extract_name)
    classy_df["superclass"] = json_df["superclass"].apply(extract_name)
    classy_df["class"] = json_df["class"].apply(extract_name)
    classy_df["subclass"] = json_df["subclass"].apply(extract_name)
    classy_df["intermediate_nodes"] = json_df["intermediate_nodes"].apply(extract_names_array)
    classy_df["alternative_parents"] = json_df["alternative_parents"].apply(extract_names_array)
    classy_df["direct_parent"] = json_df["direct_parent"].apply(extract_name)
    classy_df["molecular_framework"] = json_df["molecular_framework"]
    classy_df["substituents"] = json_df["substituents"].apply(join)
    classy_df["description"] = json_df["description"]
    classy_df["external_descriptors"] = json_df["external_descriptors"].apply(extract_external_descriptors)
    classy_df["ancestors"] = json_df["ancestors"].apply(join)
    classy_df["predicted_chebi_terms"] = json_df["predicted_chebi_terms"].apply(join)
    classy_df["predicted_lipidmaps_terms"] = json_df["predicted_lipidmaps_terms"].apply(join)
    classy_df["classification_version"] = json_df["classification_version"]

    # add suffix to all columns
    classy_df = classy_df.add_suffix(CLASSYFIRE_SUFFIX)

    return original_df.join(classy_df)


def extract_external_descriptors(json_array):
    # [{'source': 'CHEBI', 'source_id': 'CHEBI:48565', 'annotations': ['organic heteropentacyclic compound',
    # 'methyl ester', 'yohimban alkaloid']}]
    return join(["{} ({}):{}".format(json["source"], json["source_id"], json["annotations"]) for json in json_array])


def extract_name(json):
    return json["name"] if json is not None else None


def join(json_array, sep=";"):
    return sep.join(json_array)


def join(json, field, sep=";"):
    if json is None: return None
    return sep.join(json[field])


def extract_names_array(json_array):
    return ",".join([json["name"] for json in json_array if json is not None])


def search_chembl(inchi_key):
    molecule = new_client.molecule
    mol = molecule.filter(molecule_structures__standard_inchi_key='BSYNRYMUTXBXSQ-UHFFFAOYSA-N').only(
        ['molecule_chembl_id', 'pref_name'])
    logger.info(mol)

if __name__ == '__main__':
    search_chembl("")
    main()
