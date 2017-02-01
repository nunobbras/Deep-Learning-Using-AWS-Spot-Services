def wrap_with_prints(fn):

    print('wrap_with_prints runs only once')
 
    def wrapped():
        print('About to run %s' % fn.__name__)
        fn()
        print('Done running %s' % fn.__name__)

    return wrapped
 
@wrap_with_prints
def func_to_decorate():
    print('Running the function that was decorated.')
 
func_to_decorate()