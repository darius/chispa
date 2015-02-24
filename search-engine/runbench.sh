#!/bin/bash
set -euo pipefail; IFS=$'\n\t'  # 'strict mode': e=errexit, u=nounset

# You have to set stage by hand before a run;
# and also (where X = stage)
# cp -i interim_chispa.py benchX.py
# and set debug=0 in the new .py

stage=benchA

(
    echo
    echo
    date
    echo

    (
        cd hoards
        rm -rf copy-of-gutensf
        cp -r gutensf copy-of-gutensf
    )

    ./${stage}.py new ${stage}

    echo "# /usr/bin/time ./${stage}.py index ${stage} hoards/copy-of-gutensf/"
    /usr/bin/time ./${stage}.py index ${stage} hoards/copy-of-gutensf/

    echo "# du ${stage}"
    du ${stage}

    echo "# /usr/bin/time ./${stage}.py find ${stage} hello | wc -l"
    /usr/bin/time ./${stage}.py find ${stage} hello | wc -l

    echo "# /usr/bin/time ./${stage}.py find ${stage} hello meaning hola in the tongue spoken in Spain hurrah"
    /usr/bin/time ./${stage}.py find ${stage} hello meaning hola in the tongue spoken in Spain hurrah

    (
        echo "# rm copy-of-gutensf/etext03/ulyss12.txt"
        echo "# Adding 'Spain hurrah' to copy-of-gutensf/etext93/scarp10.txt"
        cd hoards
        rm copy-of-gutensf/etext03/ulyss12.txt
        echo Spain hurrah >>copy-of-gutensf/etext93/scarp10.txt
    )

    echo "# /usr/bin/time ./${stage}.py index ${stage} hoards/copy-of-gutensf/"
    /usr/bin/time ./${stage}.py index ${stage} hoards/copy-of-gutensf/

    echo "# /usr/bin/time ./${stage}.py find ${stage} hello meaning hola in the tongue spoken in Spain hurrah"
    /usr/bin/time ./${stage}.py find ${stage} hello meaning hola in the tongue spoken in Spain hurrah

) >>benchresults 2>&1
