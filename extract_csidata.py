#!/usr/bin/env python3.7
# -*-coding:utf-8-*-
# Copyright 2021 Espressif Systems (Shanghai) PTE LTD
#
# Updated by HuangZ for windows10
# Fixed bug:
# 1.Blank line problem when exporting csv
# 2.Removed csi_data other than the frequency response length of 384(Line:102)
# You can adjust to your own data

import sys
from os import path
from io import StringIO
import argparse
import csv
import json


COLUMN = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data"]


def extract_csi_data(strings: str, num: int):
    """ extract csi data from string beginning with "CSI_DATA"

    :strings: string beginning with "CSI_DATA"
    :num: the elements number in strings
    :returns: List[type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data]

    """
    f = StringIO(strings)
    csv_reader = csv.reader(f)
    csi_data = next(csv_reader)

    if len(csi_data) != num:
        raise ValueError(f"element number is not equal {num}")

    try:
        json.loads(csi_data[-1])
    except json.JSONDecodeError:
        raise ValueError(f"data is not incomplete")

    return csi_data


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="filter CSI_DATA from console_test logfile")
    parser.add_argument('-S', '--src', dest='src_file', action='store', required=True,
                        help="console_test logfile")
    parser.add_argument('-D', '--dst', dest='dst_file', action='store', default=None,
                        help="output file saved csi data")
    args = parser.parse_args()

    src_file = args.src_file
    dst_file = args.dst_file
    if dst_file is None:
        dst_file = path.splitext(path.basename(src_file))[0] + ".csv"

    f_src = open(src_file, 'r')
    f_dst = open(dst_file, 'w', newline='')
    csv_writer = csv.writer(f_dst)
    csv_writer.writerow(COLUMN)

    while True:
        strings = f_src.readline()
        if not strings:
            break

        index = strings.find(('CSI_DATA'))
        if index != -1:
            try:
                row_csi = extract_csi_data(strings[index:], 25)
                if row_csi[22] == '384':
                    csv_writer.writerow(row_csi)
                # print(row_csi)
                else:
                    continue
            except ValueError:
                continue

    f_src.close()
    f_dst.close()
    print(f"Saved csi data to {dst_file}")
