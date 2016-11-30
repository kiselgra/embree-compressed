// ======================================================================== //
// Copyright 2009-2015 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "../common/tutorial/tutorial.h"
#include "../common/tutorial/obj_loader.h"
#include "../common/tutorial/xml_loader.h"
#include "../common/image/image.h"

#include "viewer.h"

namespace embree
{
  /* name of the tutorial */
  const char* tutorialName = "viewer";

  /* configuration */
  static std::string g_rtcore = "";
  static size_t g_numThreads = 0;
  static std::string g_subdiv_mode = "";

  /* output settings */
  static size_t g_width = 512;
  static size_t g_height = 512;
  static bool g_fullscreen = false;
  static FileName outFilename = "";
  static int g_skipBenchmarkFrames = 0;
  static int g_numBenchmarkFrames = 0;
  static bool g_interactive = true;
  static bool g_anim_mode = false;
  static bool g_loop_mode = false;
  static FileName keyframeList = "";

  static std::vector<Vec3fa> cam_from;
  static std::vector<Vec3fa> cam_to;
  static std::vector<Vec3fa> cam_up;

  size_t width = g_width;
  size_t height = g_height;
  Vec3fa* orgs;
  Vec3fa* dirs;


  /* scene */
  OBJScene g_obj_scene;
  static FileName filename = "";

  static void parseCommandLine(Ref<ParseStream> cin, const FileName& path)
  {
    while (true)
    {
      std::string tag = cin->getString();
      if (tag == "") return;

      /* parse command line parameters from a file */
      else if (tag == "-c") {
        FileName file = path + cin->getFileName();
        parseCommandLine(new ParseStream(new LineCommentFilter(file, "#")), file.path());
      }

      /* load OBJ model*/
      else if (tag == "-i") {
        filename = path + cin->getFileName();
      }

      /* parse camera parameters */
      else if (tag == "-vp") g_camera.from = cin->getVec3fa();
      else if (tag == "-vi") g_camera.to = cin->getVec3fa();
      else if (tag == "-vd") g_camera.to = g_camera.from + cin->getVec3fa();
      else if (tag == "-vu") g_camera.up = cin->getVec3fa();
      else if (tag == "-fov") g_camera.fov = cin->getFloat();

	  /* parse bench camera parameters */
	  else if (tag == "-cp") cam_from.push_back(cin->getVec3fa());
	  else if (tag == "-ci") cam_to.push_back(cin->getVec3fa());
	  else if (tag == "-cd") cam_to.push_back(cam_from.back() + cin->getVec3fa());
	  else if (tag == "-cu") cam_up.push_back(cin->getVec3fa());

      /* frame buffer size */
      else if (tag == "-size") {
        width = g_width = cin->getInt();
        height = g_height = cin->getInt();
      }

      /* full screen mode */
      else if (tag == "-fullscreen") 
        g_fullscreen = true;

      /* output filename */
      else if (tag == "-o") {
        outFilename = cin->getFileName();
	g_interactive = false;
      }

      else if (tag == "-objlist") {
        keyframeList = cin->getFileName();
      }

      /* subdivision mode */
      else if (tag == "-cache") 
	g_subdiv_mode = ",subdiv_accel=bvh4.subdivpatch1cached";

      else if (tag == "-pregenerate") 
	g_subdiv_mode = ",subdiv_accel=bvh4.grid.eager";

	  else if (tag == "-oriented.ref")
    g_subdiv_mode = ",subdiv_accel=oriented.ref";
	  else if (tag == "-oriented.uni332non")
    g_subdiv_mode = ",subdiv_accel=oriented.quantizeduni";
	  else if (tag == "-oriented.pre332non")
    g_subdiv_mode = ",subdiv_accel=oriented.quantizednon";
	  else if (tag == "-oriented.uni332a")
    g_subdiv_mode = ",subdiv_accel=oriented.compresseduni";
	  else if (tag == "-oriented.pre332a")
    g_subdiv_mode = ",subdiv_accel=oriented.compressednon";
	  else if (tag == "-oriented.uni332b")
    g_subdiv_mode = ",subdiv_accel=oriented.halfslabuni";
	  else if (tag == "-oriented.pre332b")
    g_subdiv_mode = ",subdiv_accel=oriented.halfslabnon";

      else if (tag == "-loop") 
	g_loop_mode = true;

      else if (tag == "-anim") 
	g_anim_mode = true;

      /* number of frames to render in benchmark mode */
      else if (tag == "-benchmark") {
        g_skipBenchmarkFrames = cin->getInt();
        g_numBenchmarkFrames  = cin->getInt();
	g_interactive = false;
      }

      /* rtcore configuration */
      else if (tag == "-rtcore")
        g_rtcore = cin->getString();

      /* number of threads to use */
      else if (tag == "-threads")
        g_numThreads = cin->getInt();

       /* ambient light source */
      else if (tag == "-ambientlight") 
      {
        const Vec3fa L = cin->getVec3fa();
        g_obj_scene.ambientLights.push_back(OBJScene::AmbientLight(L));
      }

      /* point light source */
      else if (tag == "-pointlight") 
      {
        const Vec3fa P = cin->getVec3fa();
        const Vec3fa I = cin->getVec3fa();
        g_obj_scene.pointLights.push_back(OBJScene::PointLight(P,I));
      }

      /* directional light source */
      else if (tag == "-directionallight" || tag == "-dirlight") 
      {
        const Vec3fa D = cin->getVec3fa();
        const Vec3fa E = cin->getVec3fa();
        g_obj_scene.directionalLights.push_back(OBJScene::DirectionalLight(D,E));
      }

      /* distant light source */
      else if (tag == "-distantlight") 
      {
        const Vec3fa D = cin->getVec3fa();
        const Vec3fa L = cin->getVec3fa();
        const float halfAngle = cin->getFloat();
        g_obj_scene.distantLights.push_back(OBJScene::DistantLight(D,L,halfAngle));
      }

      /* skip unknown command line parameter */
      else {
        std::cerr << "unknown command line parameter: " << tag << " ";
        while (cin->peek() != "" && cin->peek()[0] != '-') std::cerr << cin->getString() << " ";
        std::cerr << std::endl;
      }
    }
  }
  
  void renderBenchmark(const FileName& fileName)
  {
#ifdef INCOHERENT_BENCH
	orgs = static_cast<Vec3fa*>(malloc(sizeof(Vec3fa)*g_height*g_width));
	dirs = static_cast<Vec3fa*>(malloc(sizeof(Vec3fa)*g_height*g_width));
#endif

    resize(g_width,g_height);

	//for (size_t k = 0; k < cam_from.size() && k < cam_to.size() && k < cam_up.size(); ++k) {
	size_t k = 0;
	while (true) {

		if (k < cam_from.size() && k < cam_to.size() && k < cam_up.size()) {
		g_camera.from = cam_from[k];
		g_camera.to = cam_to[k];
		g_camera.up = cam_up[k];
		g_camera.fov = 45.f;
		std::cout << "CAM_POS: " << k << std::endl;
		++k;
		}
		else if (k > 0)
			break;


    AffineSpace3fa pixel2world = g_camera.pixel2world(g_width,g_height);



    double dt = 0.0f;
    size_t numTotalFrames = g_skipBenchmarkFrames + g_numBenchmarkFrames;
    for (size_t i=0; i<numTotalFrames; i++) 
    {

      double t0 = getSeconds();
      render(0.0f,pixel2world.l.vx,pixel2world.l.vy,pixel2world.l.vz,pixel2world.p);
      double t1 = getSeconds();
      std::cout << "frame [" << i << " / " << numTotalFrames << "] ";
      std::cout << 1.0/(t1-t0) << "fps ";
      if (i < g_skipBenchmarkFrames) std::cout << "(skipped)";
      std::cout << std::endl;
      if (i >= g_skipBenchmarkFrames) dt += t1-t0;
    }
    std::cout << "frame [" << g_skipBenchmarkFrames << " - " << numTotalFrames << "] " << std::flush;
    std::cout << double(g_numBenchmarkFrames)/dt << "fps " << std::endl;
    std::cout << "BENCHMARK_RENDER " << double(g_numBenchmarkFrames)/dt << std::endl;

	if (cam_from.size() == 0 || cam_to.size() == 0 || cam_up.size()  == 0)
		break;

	}

#ifdef INCOHERENT_BENCH
	free(orgs);
	free(dirs);
#endif
  }


  void renderToFile(const FileName& fileName)
  {
    resize(g_width,g_height);
    AffineSpace3fa pixel2world = g_camera.pixel2world(g_width,g_height);
    render(0.0f,pixel2world.l.vx,pixel2world.l.vy,pixel2world.l.vz,pixel2world.p);
    void* ptr = map();
    Ref<Image> image = new Image4uc(g_width, g_height, (Col4uc*)ptr);
    storeImage(image, fileName);
    unmap();
    cleanup();
  }

  /* main function in embree namespace */
  int main(int argc, char** argv) 
  {
    /* for best performance set FTZ and DAZ flags in MXCSR control and status register */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    /* create stream for parsing */
    Ref<ParseStream> stream = new ParseStream(new CommandLineStream(argc, argv));

    /* parse command line */  
    parseCommandLine(stream, FileName());

    /* load default scene if none specified */
    if (filename.ext() == "") {
      FileName file = FileName::executableFolder() + FileName("models/cornell_box.ecs");
      parseCommandLine(new ParseStream(new LineCommentFilter(file, "#")), file.path());
    }

    /* configure number of threads */
    if (g_numThreads) 
      g_rtcore += ",threads=" + std::to_string((long long)g_numThreads);
    if (g_numBenchmarkFrames)
      g_rtcore += ",benchmark=1";

    g_rtcore += g_subdiv_mode;

    /* load scene */
    if (strlwr(filename.ext()) == std::string("obj")) {
      if (g_subdiv_mode != "") {
        std::cout << "enabling subdiv mode" << std::endl;
        loadOBJ(filename,one,g_obj_scene,true);	
      }
      else
        loadOBJ(filename,one,g_obj_scene);
    }
    else if (strlwr(filename.ext()) == std::string("xml"))
      loadXML(filename,one,g_obj_scene);
    else if (filename.ext() != "")
      THROW_RUNTIME_ERROR("invalid scene type: "+strlwr(filename.ext()));

    /* initialize ray tracing core */
    init(g_rtcore.c_str());

    /* send model */
    set_scene(&g_obj_scene);
    
    /* benchmark mode */
    if (g_numBenchmarkFrames)
      renderBenchmark(outFilename);
    
    /* render to disk */
    if (outFilename.str() != "")
      renderToFile(outFilename);
    
    /* interactive mode */
    if (g_interactive) {
      initWindowState(argc,argv,tutorialName, g_width, g_height, g_fullscreen);
      enterWindowRunLoop(g_anim_mode);
    }

    return 0;
  }
}

int main(int argc, char** argv)
{
  try {
    return embree::main(argc, argv);
  }
  catch (const std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cout << "Error: unknown exception caught." << std::endl;
    return 1;
  }
}
