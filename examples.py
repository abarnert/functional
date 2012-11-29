"""
This file contains examples of functional programming in action.
"""

### sum, implemented in terms of foldl
################################################################
from functional import foldl

try:
    from operator import add
except ImportError:
    def add(a, b):
        return a + b

def sum(itr):
    """
    sum(iterable) -> number
    
    sum([x0, x1, x2,..., xN]) is equivalent to 0 + x0 + x1 + x2 + ... + xN
    """
    return foldl(add, 0, itr)

### product, implemented in terms of foldl
################################################################

try:
    from operator import mul
except ImportError:
    def mul(a, b):
        return a * b

def product(itr):
    """
    product(iterable) -> number
    
    product([x0, x1, x2,..., xN]) is equivalent to 1 * x0 * x1 * x2 * ... * xN
    """

    return foldl(mul, 1, itr)

### Using compose, partial and map
################################################################
from functional import compose, partial

class MyClass(object):
    def __init__(self, foo, bar, baz):
        self.foo = foo
        self.bar = bar
        self.baz = baz
        
    def __hash__(self):
        attrs = ['__class__', 'foo', 'bar', 'baz']
        self_getattr = partial(getattr, self)
        str_hash_getattr = compose(str, compose(hash, self_getattr))
        
        # In Haskell, we could use concatMap here, since the String type
        # just an alias for a list of Chars. In Python, strings may obey
        # the sequence protocol, but that doesn't mean we can do
        # "abc" + []
        return hash(''.join(map(str_hash_getattr, attrs)))
