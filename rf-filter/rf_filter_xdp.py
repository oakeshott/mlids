#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from bcc import BPF
from bcc import lib
from pyroute2 import IPRoute
import sys
import time
from socket import inet_ntop, ntohs, AF_INET, AF_INET6
from struct import pack
import ctypes as ct
import json
import numpy as np
from datetime import datetime

curdir = os.path.dirname(__file__)
ipr = IPRoute()

bpf_text = """
#include <uapi/linux/bpf.h>
#include <linux/inet.h>
#include <linux/pkt_cls.h>
#include <linux/ip.h>
#include <uapi/linux/tcp.h>
#include <uapi/linux/udp.h>

#define DEBUG_BUILD
#define TREE_LEAF -1
#define TREE_UNDEFINED -2
#define MAX_TREE_DEPTH 20
#define FIXED_POINT_DIGITS 16
#define NUM_FEATURES 12
#define OUTPUT_DIM 2
#define NUM_ESTIMATORS N_ESTIMATORS
#define M MAX_PARAM_LENGTH
#ifndef abs
#define abs(x) ((x)<0 ? -(x) : (x))
#endif

struct pkt_t {
  int64_t sport;
  int64_t dport;
  int64_t protocol;
  int64_t tot_len;
  int64_t interval_time;
  int64_t direction;
  int64_t last_packet_timestamp;
  int64_t is_anomaly;
};

struct pkt_key_t {
  u32 protocol;
  u32 saddr;
  u32 daddr;
  u32 sport;
  u32 dport;
};

struct pkt_leaf_t {
  u64 num_packets;
  u64 last_packet_timestamp;
  u64 sport;
  u64 dport;
  u64 features[6];
  bool is_anomaly;
};


BPF_TABLE("lru_hash", struct pkt_key_t, struct pkt_leaf_t, sessions, 1024);
BPF_HASH(dropcnt, int, u32);
BPF_ARRAY(childrenLeft, int64_t, DT_CHILDREN_LEFT_SIZE);
BPF_ARRAY(childrenRight, int64_t, DT_CHILDREN_RIGHT_SIZE);
BPF_ARRAY(feature, int64_t, DT_FEATURE_SIZE);
BPF_ARRAY(threshold, int64_t, DT_THRESHOLD_SIZE);
BPF_ARRAY(value, int64_t, DT_VALUE_SIZE);

int rf_xdp_drop_packet(struct xdp_md *ctx) {
  int64_t ts = bpf_ktime_get_ns();
  void* data_end = (void*)(long)ctx->data_end;
  void* data = (void*)(long)ctx->data;

  struct ethhdr *eth = data;
  u64 nh_off = sizeof(*eth);

  struct iphdr *iph;
  struct tcphdr *th;
  struct udphdr *uh;
  struct pkt_key_t pkt_key = {};
  struct pkt_leaf_t pkt_val = {};


  pkt_key.protocol = 0;
  pkt_key.saddr = 0;
  pkt_key.daddr = 0;
  pkt_key.sport = 0;
  pkt_key.dport = 0;

  ethernet: {
    if (data + nh_off > data_end) {
      return XDP_DROP;
    }
    switch(eth->h_proto) {
      case htons(ETH_P_IP): goto ip;
      default: goto EOP;
    }
  }

  ip: {
    iph = data + nh_off;
    if ((void*)&iph[1] > data_end)
      return XDP_DROP;
    pkt_key.saddr    = iph->saddr;
    pkt_key.daddr    = iph->daddr;
    pkt_key.protocol = iph->protocol;

    switch(iph->protocol) {
      case IPPROTO_TCP: goto tcp;
      case IPPROTO_UDP: goto udp;
      default: goto EOP;
    }
  }

  tcp: {
    th = (struct tcphdr *)(iph + 1);
    if ((void*)(th + 1) > data_end) {
      return XDP_DROP;
    }
    pkt_key.sport = ntohs(th->source);
    pkt_key.dport = ntohs(th->dest);

    goto rf;
  }

  udp: {
    uh = (struct udphdr *)(iph + 1);
    if ((void*)(uh + 1) > data_end) {
      return XDP_DROP;
    }
    pkt_key.sport = ntohs(uh->source);
    pkt_key.dport = ntohs(uh->dest);

    goto rf;
  }

  rf: {
    struct pkt_leaf_t *pkt_leaf = sessions.lookup(&pkt_key);
    if (!pkt_leaf) {
      struct pkt_leaf_t zero = {};
      zero.sport = pkt_key.sport;
      zero.dport = pkt_key.dport;
      zero.num_packets = 0;
      zero.last_packet_timestamp = ts;
      sessions.lookup_or_try_init(&pkt_key, &zero);
      pkt_leaf = sessions.lookup(&pkt_key);
    }
    if (pkt_leaf != NULL) {
      pkt_leaf->num_packets += 1;
      int64_t sport = pkt_leaf->sport;
      int64_t dport = pkt_leaf->dport;
      int64_t protocol = iph->protocol;
      int64_t tot_len = ntohs(iph->tot_len);
      int64_t interval_time = 0;
      if (pkt_leaf->last_packet_timestamp > 0) {
        interval_time = ts - pkt_leaf->last_packet_timestamp;
      }
      pkt_leaf->last_packet_timestamp = ts;
      int64_t direction = pkt_key.sport == sport;
      struct pkt_t pkt = {sport, dport, protocol, tot_len, interval_time, direction, pkt_leaf->last_packet_timestamp};

      sport <<= FIXED_POINT_DIGITS;
      dport <<= FIXED_POINT_DIGITS;
      protocol <<= FIXED_POINT_DIGITS;
      tot_len <<= FIXED_POINT_DIGITS;
      interval_time <<= FIXED_POINT_DIGITS;
      direction <<= FIXED_POINT_DIGITS;

      pkt_leaf->features[0] += tot_len;
      pkt_leaf->features[1] += interval_time;
      pkt_leaf->features[2] += direction;

      int64_t avg_tot_len       = pkt_leaf->features[0]/pkt_leaf->num_packets;
      int64_t avg_interval_time = pkt_leaf->features[1]/pkt_leaf->num_packets;
      int64_t avg_direction     = pkt_leaf->features[2]/pkt_leaf->num_packets;

      pkt_leaf->features[3] += abs(tot_len - avg_tot_len);
      pkt_leaf->features[4] += abs(interval_time - avg_interval_time);
      pkt_leaf->features[5] += abs(direction - avg_direction);

      int64_t avg_dev_tot_len       = pkt_leaf->features[3]/pkt_leaf->num_packets;
      int64_t avg_dev_interval_time = pkt_leaf->features[4]/pkt_leaf->num_packets;
      int64_t avg_dev_direction     = pkt_leaf->features[5]/pkt_leaf->num_packets;

      int64_t feat[NUM_FEATURES] = {sport, dport, protocol, tot_len, interval_time, direction, avg_tot_len, avg_interval_time, avg_direction, avg_dev_tot_len, avg_dev_interval_time, avg_dev_direction};

      int i, j, current_node;
      int64_t _zero = 0;
      unsigned int m;
      int accumulator[OUTPUT_DIM] = {0};
      int argmax = 0;
      int max_value = 0;
      int64_t current_left_child, current_right_child, current_feature, current_threshold, current_feature_value, current_value;


      for (j = 0; j < NUM_ESTIMATORS; j++) {
        current_node = 0;
        m = j*M + current_node;
        for (i = 0; i < MAX_TREE_DEPTH; i++) {
          current_left_child  = *childrenLeft.lookup_or_init(&m, &_zero);
          current_right_child = *childrenRight.lookup_or_init(&m, &_zero);
          current_feature     = *feature.lookup_or_init(&m, &_zero);
          current_threshold   = *threshold.lookup_or_init(&m, &_zero);
          if (current_right_child == TREE_LEAF || current_left_child == TREE_LEAF || current_threshold == TREE_UNDEFINED || current_feature == TREE_UNDEFINED) {
            break;
          } else {
            if (current_feature >= 0 && current_feature < NUM_FEATURES) {
              current_feature_value = feat[current_feature];
              if (current_feature_value) {
                if (current_feature_value <= current_threshold) {
                  current_node = (int) current_left_child;
                } else {
                  current_node = (int) current_right_child;
                }
              }
            }
          }
        }
        if (current_node != -1) {
          current_value = *value.lookup_or_init(&m, &_zero);
          if (current_value >= 0 && current_value < OUTPUT_DIM) {
            accumulator[current_value]++;
          }
        }
      }
      for (j = 0; j < OUTPUT_DIM; j++) {
        if (accumulator[j] > max_value) {
          argmax = j;
          max_value = accumulator[j];
        }
      }
      int _z = 0;
      u32 val = 0, *vp;
      vp = dropcnt.lookup_or_init(&_z, &val);
      *vp += 1;
      if (max_value) {
        return XDP_PASS;
        // return XDP_DROP;
      }
    }
  }

  EOP: {
    return XDP_PASS;
  }

  return XDP_PASS;
}
"""

def map_bpf_table(hashmap, values):
    MAP_SIZE = len(values)
    assert len(hashmap.items()) == MAP_SIZE
    keys = (hashmap.Key * MAP_SIZE)()
    new_values = (hashmap.Leaf * MAP_SIZE)()

    for i in range(MAP_SIZE):
        keys[i] = ct.c_int(i)
        new_values[i] = ct.c_longlong(values[i])
    hashmap.items_update_batch(keys, new_values)

if __name__ == '__main__':

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        usage()
    device = sys.argv[1]
    resdir = sys.argv[2]
    flags = 0
    offload_device = None
    if len(sys.argv) == 4:
        if "-S" in sys.argv:
            # XDP_FLAGS_SKB_MODE
            flags |= BPF.XDP_FLAGS_SKB_MODE
        if "-D" in sys.argv:
            # XDP_FLAGS_DRV_MODE
            flags |= BPF.XDP_FLAGS_DRV_MODE
        if "-H" in sys.argv:
            # XDP_FLAGS_HW_MODE
            offload_device = device
            flags |= BPF.XDP_FLAGS_HW_MODE
    prefix_path = f"{curdir}/runs"

    # bpf_maps = [f"""
    # BPF_ARRAY(childrenLeft{i}, int64_t, DT{i}_CHILDREN_LEFT_SIZE);
    # BPF_ARRAY(childrenRight{i}, int64_t, DT{i}_CHILDREN_RIGHT_SIZE);
    # BPF_ARRAY(feature{i}, int64_t, DT{i}_FEATURE_SIZE);
    # BPF_ARRAY(threshold{i}, int64_t, DT{i}_THRESHOLD_SIZE);
    # BPF_ARRAY(value{i}, int64_t, DT{i}_VALUE_SIZE);
    # """ for i in range(num_estimators)]
    #
    #
    # bpf_text = bpf_text.replace('BPF_MAP_INFO', f"{''.join(bpf_maps)}")
    # for i in range(num_estimators):
    #     bpf_text = bpf_text.replace(f"DT{i}_CHILDREN_LEFT_SIZE", f"{len(children_left)}")
    #     bpf_text = bpf_text.replace(f"DT{i}_CHILDREN_RIGHT_SIZE", f"{len(children_right)}")
    #     bpf_text = bpf_text.replace(f"DT{i}_FEATURE_SIZE", f"{len(feature)}")
    #     bpf_text = bpf_text.replace(f"DT{i}_VALUE_SIZE", f"{len(value)}")
    #     bpf_text = bpf_text.replace(f"DT{i}_THRESHOLD_SIZE", f"{len(threshold)}")
    #     bpf_text = bpf_text.replace(f"DT{i}_MAX_TREE_DEPTH", f"20")
    device = sys.argv[1]

    num_estimators = 10
    children_left = []
    children_right = []
    threshold = []
    feature = []
    value = []
    for i in range(0, num_estimators):
        with open(f'{prefix_path}/childrenLeft{i}', 'r') as f:
            children_left.append(json.load(f))
        with open(f'{prefix_path}/childrenRight{i}', 'r') as f:
            children_right.append(json.load(f))
        with open(f'{prefix_path}/threshold{i}', 'r') as f:
            threshold.append(json.load(f))
        with open(f'{prefix_path}/feature{i}', 'r') as f:
            feature.append(json.load(f))
        with open(f'{prefix_path}/value{i}', 'r') as f:
            value.append(json.load(f))
    max_children_left_size  = max([len(children_left[i]) for i in range(num_estimators)])
    max_children_right_size = max([len(children_right[i]) for i in range(num_estimators)])
    max_feature_size        = max([len(feature[i]) for i in range(num_estimators)])
    max_value_size          = max([len(value[i]) for i in range(num_estimators)])
    max_threshold_size      = max([len(threshold[i]) for i in range(num_estimators)])
    for i in range(num_estimators):
        if len(value[i]) < max_value_size:
            tmp = max_value_size - len(value[i])
            value[i] += [0] * tmp
        if len(children_left[i]) < max_children_left_size:
            tmp = max_children_left_size - len(children_left[i])
            children_left[i] += [-1] * tmp
        if len(children_right[i]) < max_children_right_size:
            tmp = max_children_right_size - len(children_right[i])
            children_right[i] += [-1] * tmp
        if len(threshold[i]) < max_threshold_size:
            tmp = max_threshold_size - len(threshold[i])
            threshold[i] += [-2] * tmp
        if len(feature[i]) < max_feature_size:
            tmp = max_feature_size - len(feature[i])
            feature[i] += [-2] * tmp

    children_right = np.array(children_right).ravel()
    children_left  = np.array(children_left).ravel()
    threshold      = np.array(threshold).ravel()
    feature        = np.array(feature).ravel()
    value          = np.array(value).ravel()

    bpf_text = bpf_text.replace("MAX_PARAM_LENGTH", f"{max_children_right_size}")
    bpf_text = bpf_text.replace("N_ESTIMATORS", f"{num_estimators}")
    bpf_text = bpf_text.replace('DT_CHILDREN_LEFT_SIZE', f"{len(children_left)}")
    bpf_text = bpf_text.replace('DT_CHILDREN_RIGHT_SIZE', f"{len(children_right)}")
    bpf_text = bpf_text.replace('DT_FEATURE_SIZE', f"{len(feature)}")
    bpf_text = bpf_text.replace('DT_VALUE_SIZE', f"{len(value)}")
    bpf_text = bpf_text.replace('DT_THRESHOLD_SIZE', f"{len(threshold)}")
    bpf_text = bpf_text.replace('DT_MAX_TREE_DEPTH', f"20")

    INGRESS = "ffff:ffff2"
    EGRESS = "ffff:ffff3"

    ret = []
    try:
        b = BPF(text=bpf_text, debug=0)
        b.attach_xdp(device, fn = b.load_func("rf_xdp_drop_packet", BPF.XDP), flags=flags)
        idx = ipr.link_lookup(ifname=device)[0]

        # ipr.tc("add", "clsact", idx);
        # ipr.tc("add-filter", "bpf", idx, ":1", fd=fn.fd, name=fn.name, parent=INGRESS, classid=1, direct_action=True)
        # for i in range(0, lib.bpf_num_functions(b.module)):
        #     func_name = lib.bpf_function_name(b.module, i)
        #     print(func_name, lib.bpf_function_size(b.module, func_name))

        dropcnt  = b.get_table("dropcnt")

        map_children_right = b.get_table("childrenRight")
        map_children_left  = b.get_table("childrenLeft")
        map_value          = b.get_table("value")
        map_threshold      = b.get_table("threshold")
        map_feature        = b.get_table("feature")

        map_bpf_table(map_children_right, children_right)
        map_bpf_table(map_children_left, children_left)
        map_bpf_table(map_value, value)
        map_bpf_table(map_threshold, threshold)
        map_bpf_table(map_feature, feature)
        interval = 110
        start = datetime.now()
        while True:
            try:
                dropcnt.clear()
                start1 = datetime.now()
                time.sleep(1)
                end = datetime.now()
                for k, v in dropcnt.items():
                    # print(v.value)
                    ret.append(int(v.value / (end - start1).total_seconds()))
                duration = (end - start).total_seconds()
                if duration > interval:
                    break
            except KeyboardInterrupt:
                break
    finally:
        filename = f"{resdir}/rxpps.log"
        if "-S" in sys.argv:
            # XDP_FLAGS_SKB_MODE
            filename = f"{resdir}/rxpps.log"
        if "-D" in sys.argv:
            filename = f"{resdir}/rxpps.log"
        with open (filename, 'w') as f:
            for d in ret:
                f.write(f"{d}\n")
        b.remove_xdp(device, flags)
