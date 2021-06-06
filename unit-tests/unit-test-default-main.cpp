// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2021 Intel Corporation. All Rights Reserved.

/*
This file is automatically added to any unit-test project generated by unit-test-config.py.

It provides the default main() for our unit-tests so that they can receive any custom options (such
as 'context').

If you don't want this file included, use this cmake directive:
    //#cmake:custom-main
Catch2 will then include its own main() by default.
If you want to create your own main(), define CATCH_CONFIG_RUNNER before including catch.h so Catch2
doesn't.
*/

// We are not using the main from catch2
#define CATCH_CONFIG_RUNNER
#include <unit-tests/catch/catch.hpp>
#include <string>

namespace test {
    std::string context;
}

using namespace Catch::clara;

int main( int argc, char * argv[] )
{
    Catch::Session session;
    // The following lines define a command line option for all tests using the flag '--context' and
    // save it's value into test::context. If you wish to have this option in your custom main you
    // need this to be in it. If you wish to add more option you can do so by adding more options
    // with "|" in between every 2 option.
    auto cli = session.cli()
             | Opt( test::context, "context" )["--context"]( "Context in which to run the tests" );

    session.cli( cli );

    auto ret = session.applyCommandLine( argc, argv );
    if( ret )
    {
        return ret;
    }

    return session.run();
}
