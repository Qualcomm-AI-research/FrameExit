# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import pprint

import yaml


class Config(object):
    """
    Config is a wrapper around YAML which turns a YAML file or a hierarchy
    of YAML file into a single namespace.

    """

    def __init__(self, dictionary):
        """
        Turns a dict into a namespace like config, recursively.

        Parameters
        ----------
        dictionary : dict
            Dictionary of configuration parameters.

        Returns
        -------
        config : hermes.training.config.Config
            Config instance.

        """
        for key, value in dictionary.items():
            key = key.replace("-", "_")
            if isinstance(value, dict):
                parsed_value = Config(value)
            else:
                parsed_value = value
            setattr(self, key, parsed_value)

    def get(self, field_name, default_value=None):
        """

        Parameters
        ----------
        field_name: str
            field name to get (possibly nested)
        default_value
            value to return if the field (or any of the subfields) does not exist.

        Returns
        -------

        """
        field_name, *subfield_names = field_name.split(".")
        if subfield_names:
            return getattr(self, field_name, Config({})).get(
                ".".join(subfield_names), default_value
            )
        return getattr(self, field_name, default_value)

    @classmethod
    def load(cls, *filepaths, **kwargs):
        """
        Load config from YAML files into a namespace
        like object called Config.

        Parameters
        ----------
        *filepaths : str or list of str
            List of filepaths. Order matters. Earlier
            config will be overwritten by later config.
        **kwargs : dict
            Use it to manually update some entries in the config
            files. Kwarg key has to exactly match one of the key
            somewhere in the nested config. If multiple exact match,
            each will be modified to the new value!


        """

        if not filepaths:
            raise ValueError("Please specify at least one or more filepaths")

        config = None
        for filepath in filepaths:
            print("Loading config @ '{}'".format(filepath))
            with open(filepath, "r") as handle:
                unparsed_config = yaml.load(handle, Loader=yaml.Loader)
            partial_config = Config(unparsed_config)
            if config is None:
                config = partial_config
            else:
                config += partial_config

        config.update(**kwargs)
        return config

    def serialize(self):
        """
        Serialize the config namespace into a
        nested dictionary.

        Returns
        -------
        dictionary : dict
            Dictionary of the config.

        """
        dictionary = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                dictionary[key] = value.serialize()
            else:
                dictionary[key] = value
        return dictionary

    def save(self, filepath, **kwargs):
        """
        Save the config to YAML file.

        Parameters
        ----------
        filepath : str
            Path where to write YAML file.

        """
        dictionary = self.serialize()
        with open(filepath, "w") as outfile:
            yaml.dump(dictionary, outfile, default_flow_style=False, **kwargs)

    def __add__(self, other):
        """
        Merge config 2 inside config 1, overwriting
        already existing key: value pair from config 1 if
        key: value pair exist in both. I repeat, config 2
        overwrites config 1.

        """
        new = copy.deepcopy(self)
        for key, value in other.__dict__.items():
            if key not in self.__dict__:
                setattr(new, key, value)
            elif isinstance(value, Config):
                # Note, this add the config `value` to self.key, and as such will invoke recursion
                setattr(new, key, getattr(self, key) + value)
            else:
                current_value = getattr(self, key)
                if current_value != value:
                    print(
                        "  Overwriting {} from {} to {}.".format(
                            key, current_value, value
                        )
                    )
                    setattr(new, key, value)
        return new

    def update(self, **kwargs):
        """
        Update is used to update a key in the config,
        anywhere in the nesting (has to be an exact match).

        Limitations:
          - If two keys would match exactly, both would be overwritten.
          - Cannot create a new key.

        """

        kwargs = parse_hierarchical_kwargs(kwargs)

        # TODO: Raise if argument exists in duplo
        for key, value in kwargs.items():
            if key in self.__dict__:
                current_value = getattr(self, key)
                if isinstance(current_value, Config):
                    if isinstance(value, dict):
                        current_value.update(**value)
                    elif isinstance(value, Config):
                        current_value.update(**value.__dict__)
                    else:
                        raise ValueError(
                            "Cannot overwrite the category {}.".format(current_value)
                        )
                else:
                    print(
                        "  Overwriting '{}' from '{}' to '{}'.".format(
                            key, current_value, value
                        )
                    )
                    setattr(self, key, value)
            else:
                updated_any = False
                for dict_value in self.__dict__.values():
                    if isinstance(dict_value, Config):
                        try:
                            dict_value.update(**{key: value})
                        except KeyError:
                            pass
                        else:
                            updated_any = True
                if not updated_any:
                    raise KeyError("'{}' is not a key in Config".format(key))

    def pformat(self, *args, **kwargs):
        dictionary = self.serialize()
        return pprint.pformat(dictionary, *args, **kwargs)

    def __repr__(self):
        cls_name = self.__class__.__name__
        prefix = cls_name + "("
        postfix = ")"
        # Format and indent the dict_str
        dict_str = self.pformat(width=80 - len(prefix))
        dict_str = ("\n" + " " * len(prefix)).join(dict_str.split("\n"))
        return prefix + dict_str + postfix

    def __eq__(self, other):
        return self.__dict__ == other

    def __contains__(self, item):
        return item in self.__dict__


def parse_hierarchical_kwargs(kwargs):
    """Parses a dict of kwargs by splitting the keys based on separator into
    subdictionaries

    Parameters
    ----------
    kwargs : dict
        the kwargs to parse

    Returns
    -------
    hierarchical_kwargs : dict

    Example
    -------
    > parse_hierarchical_kwargs({'a.b.c': 2, 'a.d': 5, 'e': 10})
    {'a': {'b': {'c': 2}, 'd': 5}, 'e': 10}


    """

    new_kwargs = {}

    for key, value in kwargs.items():
        k_split = key.split(".")
        root_k = k_split[0]
        child_k = ".".join(k_split[1:])

        if child_k:
            if root_k not in new_kwargs:
                new_kwargs[root_k] = {}

            new_kwargs[root_k][child_k] = value
        else:
            new_kwargs[key] = value

    for key in new_kwargs:
        if isinstance(new_kwargs[key], dict):
            new_kwargs[key] = parse_hierarchical_kwargs(new_kwargs[key])

    return new_kwargs


def embed_config(arg):
    return Config.load(arg)
