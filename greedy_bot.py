import interface as bbox
import theano
import numpy as np
import theano.tensor as T
import lasagne


def get_action_by_state(state):
    scores = get_all_scores(state)
    action_to_do = np.argmax(scores)
    # print (scores,action_to_do)
    # raw_input("Press Enter to continue...")
    return action_to_do

n_features = n_actions = max_time = -1

def get_all_scores(state,verbose=0):
    checkpoint_id = bbox.create_checkpoint()
    all_scores = np.zeros(shape=n_actions)
    for a in range(n_actions):
        for _ in range(100):
            bbox.do_action(a)
        all_scores[a]=bbox.get_score()
        bbox.load_from_checkpoint(checkpoint_id)
    return all_scores
 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox(verbose=False):
    has_next = 1
    
    prepare_bbox()

    while has_next:
        state = bbox.get_state()
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)
 
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 