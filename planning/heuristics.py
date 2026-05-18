from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_all_groundings


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###
    pendientes = set(goal - state)
    if not pendientes:
        return 0.0
    acciones = get_all_groundings(domain, objects)
    conteo = 0
    while pendientes:
        mejor_accion = None
        mejor_cover = frozenset()
        for accion in acciones:
            cover = accion.add_list & pendientes
            if len(cover) > len(mejor_cover):
                mejor_cover = cover
                mejor_accion = accion

        if mejor_accion is None or not mejor_cover:
            break
        
        pendientes -= mejor_cover
        conteo += 1
    
    return float(conteo)
        
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    relaxed = set(state)
    if goal.issubset(relaxed):
        return 0.0

    actions = get_all_groundings(domain, objects)

    steps = 0
    while not goal.issubset(relaxed):
        best_action = None
        best_goal_score = -1
        best_new_count = -1

        for action in actions:
            # Las precondiciones siguen vigentes en esta relajación;
            # solo se ignora la delete list.
            if not action.precond_pos.issubset(relaxed):
                continue
            if not action.precond_neg.isdisjoint(relaxed):
                continue

            added = action.add_list - relaxed   # fluentes realmente nuevos
            if not added:
                continue                         # sin progreso en modelo monótono

            goal_score = len((relaxed | action.add_list) & goal)
            if goal_score > best_goal_score or (
                goal_score == best_goal_score and len(added) > best_new_count
            ):
                best_goal_score = goal_score
                best_new_count = len(added)
                best_action = action

        if best_action is None:
            break  # nada nuevo aplicable → meta inalcanzable en la relajación

        relaxed |= best_action.add_list
        steps += 1

    return float(steps)

    ### End of your code ###
