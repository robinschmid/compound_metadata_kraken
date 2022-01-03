def json_col(result_df, json_column, suffix, field, apply_function=None, new_col_name=None):
    if new_col_name is None:
        new_col_name = field
    if apply_function is None:
        result_df[new_col_name+suffix] = json_column.apply(
            lambda json: json[field] if json and field in json else None)
    else:
        result_df[new_col_name+suffix] = json_column.apply(
            lambda json: None if json is None or field not in json else apply_function(json[field]))
    return result_df


def get_chembl_xref(xrefs, xref_src):
    xref_src = xref_src.casefold()
    xref_ids = [x["xref_id"] for x in xrefs if x["xref_src"].casefold() == xref_src]
    return ";".join(xref_ids) if xref_ids else None


def extract_external_descriptors(json_array):
    # [{'source': 'CHEBI', 'source_id': 'CHEBI:48565', 'annotations': ['organic heteropentacyclic compound',
    # 'methyl ester', 'yohimban alkaloid']}]
    return join(["{} ({}):{}".format(json["source"], json["source_id"], json["annotations"]) for json in json_array])


def extract_name(json):
    return json["name"] if json is not None else None


def join(json_array, sep=";"):
    return sep.join(json_array)


def join_by_field(json, field, sep=";"):
    if json is None: return None
    return sep.join(json[field])


def join_array_by_field(json_array, field, sep=";"):
    return sep.join([json[field] for json in json_array if json and field in json])


def extract_names_array(json_array):
    return join_array_by_field(json_array, "name")
