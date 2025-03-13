from sklearn.ensemble import RandomForestRegressor

def get_tree_paths(tree,feature_names):
    paths = []
    stack = [(0, [])]
    while stack:
        node_id, path = stack.pop()
        feature_index = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]

        if feature_index != -2:  
            feature_name = feature_names[feature_index]
            left_path = path + [(feature_name, threshold, 'L')]
            stack.append((tree.tree_.children_left[node_id], left_path))
            right_path = path + [(feature_name, threshold, 'R')]
            stack.append((tree.tree_.children_right[node_id], right_path))
        else:
            paths.append(path)

    return paths

def has_negative_value(lst):
    for item in lst:
        if item < 0:
            return True
        
    return False

def in_rule(x,rule,feature_names):
    for single in rule:
        k = feature_names.index(single[0])
        if single[2] == 'L':
            if float(x[k]) <= float(single[1]):
                continue
            else:
                return False
        else:
            if float(x[k]) > float(single[1]):
                continue
            else:
                return False
            
    return True

def Model(train_independent, train_dependent):
    model = RandomForestRegressor()
    model.fit(train_independent, train_dependent)

    return model

def merge_constraints(constraints_list):
    merged = {}
    counts = {}
    for feat, val, side in constraints_list:
        op = '<' if side == 'L' else '>'
        counts[feat] = counts.get(feat, 0) + 1
        if feat not in merged:
            merged[feat] = (op, val)
        else:
            cur_op, cur_val = merged[feat]
            if op == cur_op:
                if op == '<':
                    merged[feat] = (op, min(cur_val, val))
                else:
                    merged[feat] = (op, max(cur_val, val))
    for feat in merged:
        if counts[feat] == 1 and merged[feat][0] == '<':
            merged[feat] = ('>', merged[feat][1])
    def fmt(val):
        return int(val) if val == int(val) else val
    return ','.join(f"{feat}{op}{fmt(val)}" for feat, (op, val) in merged.items())
