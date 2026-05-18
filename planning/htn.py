from __future__ import annotations

from planning.pddl import Action, Problem, apply_action, is_applicable
from collections import deque



# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    ### Your code here ###
    # Initialize the search queue with the initial plan (top-level HLA) and index 0.
    queue = deque([([hlas[0]], 0)])

    while queue:
        plan, index = queue.popleft()
        # If the plan is fully primitive, check if it achieves the goal.
        if is_plan_primitive(plan):
            state = problem.initial_state
            for action in plan:
                if not is_applicable(state, action):
                    break  # If any action is not applicable, discard this plan.
                state = apply_action(state, action)
            else:
                # If we successfully applied all actions, check if we reached the goal.
                if problem.isGoalState(state):
                    return plan  # Found a valid plan!
        # If the plan is not fully primitive, find the first non-primitive step.
        for i, step in enumerate(plan):
            if not is_primitive(step):
                # Replace the non-primitive step with each of its refinements and add to the queue.
                for refinement in step.refinements:
                    new_plan = plan[:i] + refinement + plan[i+1:]
                    queue.append((new_plan, 0))  # Index is not used in this implementation.    
                break  # Only refine the first non-primitive step.
    return []  # No plan found after exploring all refinements. 

    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###
    robot = None
    robot_loc = None
    supplies = []
    patients = []
    med_posts = []
    adj_map = {}
    
    for fluent in problem.initial:
        pred = fluent[0]
        if pred == 'At':
            obj, loc = fluent[1], fluent[2]
            obj_str = str(obj)
            if obj_str.startswith('R') or obj_str == 'robot':
                robot = obj
                robot_loc = loc
            elif obj_str.startswith('T'):
                supplies.append((obj, loc))
            elif obj_str.startswith('S'):
                patients.append((obj, loc))
        elif pred == 'Medical Post':
            med_posts.append(fluent[1])
        elif pred == 'Adjacent':
            src, dst = fluent[1], fluent[2]
            adj_map.setdefault(src, []).append(dst)
            
    med_post_loc = med_posts[0] if med_posts else None
    nav_cache = {}
    
    def get_navigate(src, dst):
        if (src, dst) in nav_cache:
            return nav_cache[(src, dst)]
            
        nav_hla = HLA(f"Navigate({src},{dst})", [])
        nav_cache[(src, dst)] = nav_hla
        
        if src == dst:
            nav_hla.refinements.append([])
        else:
            for nxt in adj_map.get(src, []):
                move_act = Action('Move', (robot, src, nxt))
                nav_hla.refinements.append([move_act, get_navigate(nxt, dst)])
        return nav_hla

    top_hlas = []
    current_robot_loc = robot_loc
    
    for (p_obj, p_loc), (s_obj, s_loc) in zip(patients, supplies):
        
        prep_hla = HLA(f"PrepareSupplies({s_obj},{med_post_loc})", [])
        nav_to_s = get_navigate(current_robot_loc, s_loc)
        pickup_s = Action('PickUp', (robot, s_obj, s_loc))
        nav_s_to_m = get_navigate(s_loc, med_post_loc)
        setup_s = Action('SetupSupplies', (robot, s_obj, med_post_loc))
        
        prep_hla.refinements.append([nav_to_s, pickup_s, nav_s_to_m, setup_s])
        current_robot_loc = med_post_loc
        
        ext_hla = HLA(f"ExtractPatient({p_obj},{med_post_loc})", [])
        nav_to_p = get_navigate(current_robot_loc, p_loc)
        pickup_p = Action('PickUp', (robot, p_obj, p_loc))
        nav_p_to_m = get_navigate(p_loc, med_post_loc)
        
        putdown_p = Action('PutDown', (robot, p_obj, med_post_loc))
        rescue_p = Action('Rescue', (robot, p_obj, med_post_loc))
        
        ext_hla.refinements.append([nav_to_p, pickup_p, nav_p_to_m, putdown_p, rescue_p])
        
        full_miss = HLA(f"FullRescueMission({s_obj},{p_obj},{med_post_loc})", [])
        full_miss.refinements.append([prep_hla, ext_hla])
        
        top_hlas.append(full_miss)
        
    return top_hlas

    ### End of your code ###
