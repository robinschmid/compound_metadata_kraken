
def json_col(result_df, json_column, field, apply_function=None):
    if apply_function is None:
        result_df[field] = json_column.apply(lambda json: None if json is None else json[field])
    else:
        result_df[field] = json_column.apply(lambda json: None if json is None else apply_function(json[field]))
    return result_df



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


def extract_names_array(json_array):
    return ",".join([json["name"] for json in json_array if json is not None])