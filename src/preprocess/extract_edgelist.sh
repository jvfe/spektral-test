#!/bin/zsh

path="../../data/mbps_rins/"

for g in $(ls ${path}"edgelists/")
do
  awk '{print $1,$3}' "${path}edgelists/${g}" >> "${path}just_edges/${g}"
done

