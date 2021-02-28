import requests
import json
from flask import jsonify

from .request_params import request_column, request_settings
from .utils import get_data_in_format, get_input_range


def get_vessel_data(experiment_name, episode, tick):
    params = {
        "query": f"select {request_column.vessel_header.value} from {experiment_name}.vessel_details where episode='{episode}' and tick='{tick}'",
        "count": "true"
    }
    db_vessel_data = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    return jsonify(process_vessel_data(db_vessel_data, tick))


def get_acc_vessel_data(experiment_name, episode, start_tick, end_tick):
    input_range = get_input_range(start_tick, end_tick)
    params = {
        "query": f"select {request_column.vessel_header.value} from {experiment_name}.vessel_details where episode='{episode}' and tick in {input_range}",
        "count": "true"
    }
    db_vessel_data = requests.get(request_settings.request_url.value, headers=request_settings.request_header.value, params=params).json()
    return jsonify(process_vessel_data(db_vessel_data, start_tick))


def process_vessel_data(db_vessel_data, start_tick):
    with open(r"../nginx/static/config.json", "r")as mapping_file:
        cim_information = json.load(mapping_file)
        vessel_list = list(cim_information["vessels"].keys())
        vessel_info = cim_information["vessels"]
        route_list = list(cim_information["routes"].keys())
    for item in route_list:
        route_distance = cim_information["routes"][item]
        route_distance_length = len(route_distance)
        prev = 0
        route_distance[0]["distance"] = 0
        for index in range(1, route_distance_length):
            route_distance[index]["distance"] = route_distance[index]["distance"] + prev
            prev = route_distance[index]["distance"]
    original_vessel_data = get_data_in_format(db_vessel_data)

    frame_index_num = len(original_vessel_data["tick"].unique())
    if frame_index_num == 1:
        return get_single_snapshot_vessel_data(
            original_vessel_data, vessel_list, vessel_info, route_list, cim_information
        )
    else:
        acc_vessel_data = []
        for vessel_index in range(0, frame_index_num):
            cur_vessel_data = original_vessel_data[original_vessel_data["tick"] == str(vessel_index + start_tick)].copy()
            acc_vessel_data.append(
                get_single_snapshot_vessel_data(cur_vessel_data, vessel_list, vessel_info, route_list, cim_information)
            )
        return acc_vessel_data


def get_single_snapshot_vessel_data(original_vessel_data, vessel_list, vessel_info, route_list, cim_information):
    original_vessel_data["name"] = list(
        map(
            lambda x: vessel_list[int(x)],
            original_vessel_data["index"]
        )
    )
    original_vessel_data["speed"] = list(
        map(
            lambda x: vessel_info[x]['sailing']['speed'],
            original_vessel_data["name"]
        )
    )
    original_vessel_data["route name"] = list(
        map(
            lambda x: vessel_info[x]['route']['route_name'],
            original_vessel_data["name"]
        )
    )
    original_vessel_data["start port"] = list(
        map(
            lambda x: vessel_info[x]['route']['initial_port_name'],
            original_vessel_data["name"]
        )
    )
    original_vessel_data["start"] = 0
    vessel_data = original_vessel_data.to_json(orient='records')
    vessel_json_data = json.loads(vessel_data)
    output = []
    for item in route_list:
        vessel_in_output = []
        for vessel in vessel_json_data:
            if vessel["route name"] == item:
                start_port = vessel["start port"]
                route_distance_info = cim_information["routes"][item]
                for dis in route_distance_info:
                    if dis["port_name"] == start_port:
                        vessel["start"] = dis["distance"]
                vessel_in_output.append(vessel)
        output.append({"name": item, "vessel": vessel_in_output})

    return output