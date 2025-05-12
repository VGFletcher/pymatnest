I try to collect a few thoughts during the implementation of the RENS in pymatnest here. This might give you a first idea what has changed and it can serve as a guide for polishing the implementation.

New parameters:
- make_output_dir
- RE_pressures
- RE_n_swap_cycles
- RE_swap_interval

Necessary new features:
- Instead of using one global communicator, we now have one communicator per replica. They handle the internal NS logic, whereas the global communicator handles the RE between runs
- not printing everything to stdout anymore, instead each replica prints to an individual output file, only global output concerning the whole RENS run is printed to stdout

I did not retain all functionality, so a few things might be broken. These are not necessarily things that are hard to integrate into the new implementation. I just removed stuff that is not strictly needed to reduce complexity for a first proof of concept. Affected are e.g.:
- in-situ analyzers
- n_extra_data: Either I dont understand this or there is a bug in how the send and receive buffers are constructed, which only works for n_extra_data=3.

Cosmetic changes to be tackled in the future that might make the implementation a bit less clunky:
- remove MC_cell_P in favor of RE_pressures

Even though some things might seem to work already, they might need further thought:
- I did not carefully check random state yet
