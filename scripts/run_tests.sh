#!/bin/bash

source user_config.sh

echo 'Note: this script should be called from the root of the repository' >&2

#TESTS=weat1,weat2,weat3,weat3b,weat4,weat5,weat5b,weat6,weat6b,weat7,weat7b,weat8,weat8b,weat9,weat10,sent-weat1,sent-weat2,sent-weat3,sent-weat3b,sent-weat4,sent-weat5,sent-weat5b,sent-weat6,sent-weat6b,sent-weat7,sent-weat7b,sent-weat8,sent-weat8b,sent-weat9,sent-weat10,angry_black_woman_stereotype,angry_black_woman_stereotype_b,sent-angry_black_woman_stereotype,sent-angry_black_woman_stereotype_b,heilman_double_bind_competent_1,heilman_double_bind_competent_1-,heilman_double_bind_competent_one_sentence,heilman_double_bind_competent_one_word,sent-heilman_double_bind_competent_one_word,heilman_double_bind_likable_1,heilman_double_bind_likable_1-,heilman_double_bind_likable_one_sentence,heilman_double_bind_likable_one_word,sent-heilman_double_bind_likable_one_word
TESTS=weatk1,sent-weatk1
set -e

SEED=1111