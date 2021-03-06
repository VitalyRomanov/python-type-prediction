def get_cum_lens(body, as_bytes=False):
    """
    Calculate the cummulative lengths of each line with respect to the beginning of
    the function's body.
    """
    body_lines = body.split("\n")
    cum_lens = [0]
    for ind, line in enumerate(body_lines):
        cum_lens.append(len(line if not as_bytes else line.encode('utf8')) + cum_lens[-1] + 1) # +1 for new line character
    return cum_lens


def get_byte_to_char_map(unicode_string):
    """
    Generates a dictionary mapping character offsets to byte offsets for unicode_string.
    """
    response = {}
    byte_offset = 0
    for char_offset, character in enumerate(unicode_string):
        response[byte_offset] = char_offset
        byte_offset += len(character.encode('utf-8'))
    response[byte_offset] = len(unicode_string)
    return response


def to_offsets(body, entities, as_bytes=False):
    """
    Transform entity annotation format from (line, end_line, col, end_col)
    to (char_ind, end_char_ind).
    """
    cum_lens = get_cum_lens(body, as_bytes=as_bytes)

    repl = [(cum_lens[line] + start, cum_lens[end_line] + end, annotation) for
            ind, (line, end_line, start, end, annotation) in enumerate(entities)]

    if as_bytes:
        b2c = get_byte_to_char_map(body)
        repl = list(map(lambda x: (b2c[x[0]], b2c[x[1]], x[2]), repl))

    return repl


def overlap(p1, p2):
    if (p2[1] - p1[0]) * (p2[0] - p1[1]) <= 0:
        return True
    else:
        return False


def resolve_self_collision(offsets):
    no_collisions = []

    for ind_1, offset_1 in enumerate(offsets):
        # keep first
        if any(map(lambda x: overlap(offset_1, x), no_collisions)):
            pass
        else:
            no_collisions.append(offset_1)

    return no_collisions