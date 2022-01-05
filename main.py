from enum import Enum
from functools import partial
import pandas as pd
import requests
import urllib.parse
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils

from pubchempy import get_compounds, Compound

from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit.Chem.rdMolDescriptors as Desc

import json
import math
import pprint

from json_utils import *
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


def mol_log_p(df):
    return df[Columns.rdkit_mol.name].apply(Descriptors.MolLogP)


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


def num_heavy_atoms(df):
    return df[Columns.rdkit_mol.name].apply(lambda mol: Desc.CalcNumHeavyAtoms(mol))


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
    smiles = partial(canonical_smiles)
    canonical_smiles = partial(canonical_smiles)
    inchi = partial(inchi)
    inchi_key = partial(inchi_key)
    smarts = partial(smarts)
    formula = partial(mol_formula)
    exact_mass = partial(exact_mass)
    mass_defect = partial(mass_defect)
    mw = partial(mol_weight)
    mol_log_p = partial(mol_log_p)
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
    heavy_atoms = partial(num_heavy_atoms)
    aromatic_rings = partial(num_aromatic_rings)
    valenz = partial(NumValenceElectrons)

    def create_col(self, df):
        return self.__call__(df)

    def __str__(self):
        return self.namedef

    def __call__(self, *args):  # make it callable
        return self.value(*args)


NP_CLASSIFIER_URL = "https://npclassifier.ucsd.edu/classify?smiles={}"
CLASSYFIRE_URL = "https://gnps-structure.ucsd.edu/classyfire?smiles={}"
PUBCHEM_SUFFIX = "_pubchem"
CHEMBL_SUFFIX = "_chembl"
CLASSYFIRE_SUFFIX = "_classyfire"
NP_CLASSIFIER_SUFFIX = "_np_classifier"

# IDS ChEBI, PubChem, ChemSpider
# DBE
# violin plots
# NPAtlas:
# ORIGIN ORGANISM TYPE
NP_ATLAS_URL = "https://www.npatlas.org/api/v1/compounds/basicSearch?method=full&{}&threshold=0&orderby=npaid&ascending=true&limit=10"
NP_ATLAS_STRUCTURE_SEARCH_URL = "https://www.npatlas.org/api/v1/compounds/structureSearch?structure={}" \
                "&type={}&method=sim&threshold=0.9999&skip=0&limit=10&stereo=false"

def np_atlas_url_by_inchikey(inchi_key):
    return NP_ATLAS_URL.format("inchikey="+urllib.parse.quote(inchi_key))

def np_atlas_url_by_smiles(smiles):
    return NP_ATLAS_URL.format("smiles="+urllib.parse.quote(smiles))

def np_atlas_url_similar_structure(structure, type="inchikey"):
    return NP_ATLAS_URL.format(urllib.parse.quote(structure), type)

def search_np_atlas_by_inchikey_smiles(entry):
    # try inchikey - otherwise smiles
    result = get_json_response(np_atlas_url_by_inchikey(entry[Columns.inchi_key.name]), True)
    if result is None:
        result = get_json_response(np_atlas_url_by_smiles(entry[Columns.canonical_smiles.name]), True)
    # similar structure
    if result is None:
        result = get_json_response(np_atlas_url_similar_structure(entry[Columns.inchi_key.name]), True)
    if result is None:
        result = get_json_response(np_atlas_url_similar_structure(entry[Columns.canonical_smiles.name], "smiles"), True)

    return result

def search_np_atlas(df):
    npa = df.apply(lambda row: search_np_atlas_by_inchikey_smiles(row), axis=1)
    df["num_np_atlas_entries"] = npa.apply(len)
    npa = npa.apply(lambda result: result[0] if result else None)

    suffix = NP_CLASSIFIER_SUFFIX
    json_col(df, npa, suffix, "npaid")
    json_col(df, npa, suffix, "original_name")
    json_col(df, npa, suffix, "cluster_id")
    json_col(df, npa,  suffix,"node_id")
    json_col(df, npa, suffix, "original_type")
    json_col(df, npa, suffix, "original_organism")
    json_col(df, npa, suffix, "original_doi")

    return df

def get_pubchem_mol_by_inchikey(inchi_key):
    try:
        return get_compounds(inchi_key, 'inchikey')
    except Exception as e:
        logger.warning("Error during pubchem query:",e)
        return None

def add_pubchem_columns(df):
    # seems to be sorted in reverse
    compounds = get_pubchem_mol_by_inchikey(df[Columns.inchi_key.name].tolist())

    pc_dict = {compound.inchikey : compound for compound in compounds}

    pubchem = df[Columns.inchi_key.name].apply(lambda inchi_key: pc_dict[inchi_key])
    # df["num_pubchem_entries"] = pubchem.apply(len)
    pubchem = pubchem.apply(lambda result: result.to_dict() if result else None)

    suffix = PUBCHEM_SUFFIX
    json_col(df, pubchem, suffix, "cid")
    json_col(df, pubchem, suffix, "xlogp")
    json_col(df, pubchem, suffix, "atom_stereo_count")
    json_col(df, pubchem, suffix, "complexity")
    json_col(df, pubchem, suffix, "iupac_name")
    json_col(df, pubchem, suffix, "tpsa", None, "topological_polar_surface_area")  # Topological Polar Surface Area

    return df

def get_chembl_mol_by_inchikey(molecule, inchi_key):
    try:
        return molecule.filter(molecule_structures__standard_inchi_key=inchi_key)
    except Exception as e:
        logger.warning("Error during chembl query:",e)
        return None


def add_chembl_columns(df):
    # show all available key words
    # available_resources = [resource for resource in dir(new_client) if not resource.startswith('_')]
    # logger.info(available_resources)

    # chembl
    molecule = new_client.molecule

    chembl = df[Columns.inchi_key.name].apply(lambda inchikey: get_chembl_mol_by_inchikey(molecule, inchikey))
    # .only(['molecule_chembl_id', 'pref_name']))

    # add number of results column and then filter to only use first result
    df["num_chembl_entries"] = chembl.apply(len)
    chembl = chembl.apply(lambda result: None if result is None else result[0])

    logger.debug("chembl read finished")

    suffix = CHEMBL_SUFFIX
    json_col(df, chembl, suffix, "molecule_chembl_id")
    json_col(df, chembl, suffix, "pref_name")
    json_col(df, chembl, suffix, "molecule_synonyms", lambda syn: join_array_by_field(syn, "molecule_synonym"))
    json_col(df, chembl, suffix, "therapeutic_flag")
    json_col(df, chembl, suffix, "natural_product")
    json_col(df, chembl, suffix, "indication_class")
    json_col(df, chembl, suffix, "chirality")
    json_col(df, chembl, suffix, "max_phase")
    json_col(df, chembl, suffix, "cross_references", lambda xrefs: get_chembl_xref(xrefs, "Wikipedia"), "wikipedia_id")
    json_col(df, chembl, suffix, "cross_references", lambda xrefs: get_chembl_xref(xrefs, "PubChem"), "pubchem_id")
    # properties
    json_col(df, chembl, suffix, "molecule_properties", lambda prop: prop["alogp"], "alogp")
    json_col(df, chembl, suffix, "molecule_properties", lambda prop: prop["cx_logd"], "cx_logd")
    json_col(df, chembl, suffix, "molecule_properties", lambda prop: prop["cx_logp"], "cx_logp")
    json_col(df, chembl, suffix, "molecule_properties", lambda prop: prop["cx_most_apka"], "cx_most_apka")
    json_col(df, chembl, suffix, "molecule_properties", lambda prop: prop["cx_most_bpka"], "cx_most_bpka")

    return df
    # molecule.set_format('json')
    # aspirin = molecule.search('aspirin')
    # aspirin = molecule.search('CC(=O)OCC(CCC=C(C)C)=CCCC(CO)=CCCC(C)=CCO')

    # for r in aspirin:
    #     pref_name = r['pref_name']
    #     if pref_name is not None:
    #         print(pref_name)


def main(data_file, export_file, compute_rdkit=True, search_npclass=True, search_classyfire=True, search_pubchem=True, search_chembl=True):
    # create mol column and filter rows - missing mol means unparsable smiles or inchi
    try:
        original_df = import_data(data_file)

        # smiles, inchikey,... columns from rdkit
        if compute_rdkit:
            filtered_df = compute_rdkit_columns(original_df)
        else:
            filtered_df = original_df

        # read classes from gnps APIs
        if search_npclass:
            filtered_df = np_class(filtered_df)
        if search_classyfire:
            filtered_df = classyfire(filtered_df)

        # read data bases
        if search_pubchem:
            try:
                add_pubchem_columns(filtered_df)
            except Exception as e:
                logger.warning("broke up pubchem search", e)
        if search_chembl:
            try:
                add_chembl_columns(filtered_df)
            except Exception as e:
                logger.warning("broke up chembl search", e)
        # search_np_atlas(filtered_df)

    except Exception as e:
        logger.error("Error while parsing molecular structures", e)
        exit(1)

    # write in any case
    export_tsv_file(export_file, filtered_df)
    # success
    exit(0)


def export_tsv_file(export_file, filtered_df):
    filtered_df.to_csv(export_file, sep='\t', encoding='utf-8', index=False)


def compute_rdkit_columns(original_df):
    original_df[Columns.rdkit_mol.name] = Columns.rdkit_mol.create_col(original_df)
    filtered_df = original_df[original_df[Columns.rdkit_mol.name].astype(bool)]
    unparsable_rows = len(original_df) - len(filtered_df)
    if unparsable_rows > 0:
        unparsed_df = original_df[original_df[Columns.rdkit_mol.name].astype(bool) == False]
        unparsed_structures = get_original_structures(unparsed_df)
        logger.info("n=%d rows (structures) were not parsed: %s", unparsable_rows, "; ".join(unparsed_structures))
    else:
        logger.info("All row structures were parsed")
    # add new columns for chemical properties
    for col in Columns:
        if col.name not in filtered_df:
            filtered_df[col.name] = col.create_col(filtered_df)

    filtered_df.drop(columns=[Columns.rdkit_mol.name], axis=1, inplace=True)
    return filtered_df


def import_data(data_file):
    return pd.read_csv(data_file, sep="\t")


def classyfire_url(smiles):
    return CLASSYFIRE_URL.format(urllib.parse.quote(smiles))


def np_class_url(smiles):
    return NP_CLASSIFIER_URL.format(urllib.parse.quote(smiles))


def get_json_response(url, post=False):
    try:
        response = requests.post(url) if post else requests.get(url)
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
    json_col(df, result_column, NP_CLASSIFIER_SUFFIX, "class_results", join)
    json_col(df, result_column, NP_CLASSIFIER_SUFFIX, "superclass_results", join)
    json_col(df, result_column, NP_CLASSIFIER_SUFFIX, "pathway_results", join)
    json_col(df, result_column, NP_CLASSIFIER_SUFFIX, "isglycoside")
    json_col(df, result_column, NP_CLASSIFIER_SUFFIX, "fp1")
    json_col(df, result_column, NP_CLASSIFIER_SUFFIX, "fp2")
    return df


def classyfire(original_df):
    unique_smiles_dict = get_unique_canocical_smiles_dict(original_df)
    # Query classyfire on GNPS
    for smiles in unique_smiles_dict:
        unique_smiles_dict[smiles] = get_json_response(classyfire_url(smiles))

    # temp column with json results
    result_column = original_df[Columns.canonical_smiles.name].apply(lambda smiles: unique_smiles_dict[smiles])

    # extract information
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "kingdom", extract_name)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "superclass", extract_name)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "class", extract_name)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "subclass", extract_name)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "intermediate_nodes", extract_names_array)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "alternative_parents", extract_names_array)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "direct_parent", extract_name)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "molecular_framework")
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "substituents", join)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "description")
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "external_descriptors", extract_external_descriptors)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "ancestors", join)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "predicted_chebi_terms", join)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "predicted_lipidmaps_terms", join)
    json_col(original_df, result_column, CLASSYFIRE_SUFFIX, "classification_version")

    return original_df


if __name__ == '__main__':
    # main("data/all_smiles.tsv", "results/converted.tsv")
    main("results/full_table.tsv", "results/ft_pc.tsv", compute_rdkit=False, search_npclass=False,
         search_classyfire=False, search_pubchem=True, search_chembl=False)
